import argparse
import json
import os

import clip
import filelock
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from filelock import SoftFileLock
from torch.utils.data import DataLoader

from trap_models import SemanticLayoutGenerator, SiameseSemanticNetwork


os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["SOFT_FILELOCK"] = "1"
filelock.FileLock = SoftFileLock


def _compute_clip_saliency_targets(clip_model, clip_images: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
    # Text-conditioned saliency target: gradient of CLIP cosine(image, text) wrt image pixels.
    x = clip_images.detach().clone().requires_grad_(True)
    img = clip_model.encode_image(x).float()
    img = F.normalize(img, dim=-1)
    txt = F.normalize(text_embeds.detach().float(), dim=-1)
    sim = (img * txt).sum(dim=-1)
    grads = torch.autograd.grad(sim.sum(), x, retain_graph=False, create_graph=False)[0]
    sal = grads.abs().mean(dim=1, keepdim=True)
    sal = F.interpolate(sal, size=(64, 64), mode="bilinear", align_corners=False)

    b = sal.shape[0]
    flat = sal.view(b, -1)
    mins = flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    maxs = flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    sal = (sal - mins) / (maxs - mins + 1e-6)
    return sal.detach().clamp(0.0, 1.0)


def collate_fn(batch, clip_preprocess, device: str):
    images = [item["image"].convert("RGB") for item in batch]
    captions = [item["caption"] for item in batch]
    clip_images = torch.stack([clip_preprocess(img) for img in images]).to(device)
    return clip_images, captions


def _write_training_stats(save_dir: str, epoch: int, *, semantic_mean: float, distinctive_mean: float, siamese_mean: float, layout_mean: float) -> None:
    stats = {
        "epoch": int(epoch),
        "semantic_loss_mean": float(semantic_mean),
        "distinctive_loss_mean": float(distinctive_mean),
        "siamese_total_mean": float(siamese_mean),
        "layout_loss_mean": float(layout_mean),
    }
    with open(os.path.join(save_dir, "training_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--save_dir", type=str, default="./trap_weights")
    parser.add_argument("--distinct_weight", type=float, default=0.3, help="Weight for distinctive-identity anchor loss.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading external models...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    clip_model.requires_grad_(False)

    print("Initializing TRAP models...")
    siamese_network = SiameseSemanticNetwork(image_embed_dim=512, text_embed_dim=512, output_dim=512).to(device)
    layout_generator = SemanticLayoutGenerator(image_embed_dim=512, text_embed_dim=512).to(device)

    optimizer_siamese = torch.optim.AdamW(siamese_network.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer_layout = torch.optim.AdamW(layout_generator.parameters(), lr=args.lr)
    criterion_bce = nn.BCELoss()

    print("Loading Dataset...")
    dataset = load_dataset("SargeZT/coco-stuff-captioned", split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, clip_preprocess, device),
        drop_last=True,
    )

    print("Starting Training...")
    for epoch in range(args.epochs):
        siamese_network.train()
        layout_generator.train()

        total_siamese_loss = 0.0
        total_layout_loss = 0.0
        total_semantic_loss = 0.0
        total_distinctive_loss = 0.0

        for step, (clip_images, captions) in enumerate(dataloader):
            text_tokens = clip.tokenize(captions, truncate=True).to(device)
            with torch.no_grad():
                image_embeds = clip_model.encode_image(clip_images).float()
                text_embeds = clip_model.encode_text(text_tokens).float()

            gt_layout_masks = _compute_clip_saliency_targets(clip_model, clip_images, text_embeds)

            optimizer_siamese.zero_grad()
            combined_features, _, _ = siamese_network(image_embeds, text_embeds, mode="both")
            distinctive_features = siamese_network(image_embeds, mode="distinctive")
            cos_pos = F.cosine_similarity(
                F.normalize(combined_features, dim=-1),
                F.normalize(text_embeds, dim=-1),
                dim=-1,
            )
            loss_sem = (1.0 - cos_pos).mean()
            loss_distinct = F.mse_loss(
                F.normalize(distinctive_features, dim=-1),
                F.normalize(image_embeds.detach(), dim=-1),
            )
            loss_siamese = loss_sem + float(args.distinct_weight) * loss_distinct
            loss_siamese.backward()
            optimizer_siamese.step()
            total_siamese_loss += float(loss_siamese.item())
            total_semantic_loss += float(loss_sem.item())
            total_distinctive_loss += float(loss_distinct.item())

            optimizer_layout.zero_grad()
            pred_layouts = layout_generator(text_embeds.detach(), image_embeds.detach())
            loss_layout = criterion_bce(pred_layouts, gt_layout_masks)
            loss_layout.backward()
            optimizer_layout.step()
            total_layout_loss += float(loss_layout.item())

            if step % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] Step {step} | "
                    f"Semantic Loss: {loss_sem.item():.4f} | "
                    f"Distinctive Loss: {loss_distinct.item():.4f} | "
                    f"Siamese Total: {loss_siamese.item():.4f} | "
                    f"Layout Loss: {loss_layout.item():.4f}"
                )

        num_batches = max(len(dataloader), 1)
        avg_semantic = total_semantic_loss / num_batches
        avg_distinctive = total_distinctive_loss / num_batches
        avg_siamese = total_siamese_loss / num_batches
        avg_layout = total_layout_loss / num_batches
        torch.save(siamese_network.state_dict(), os.path.join(args.save_dir, f"siamese_epoch_{epoch+1}.pt"))
        torch.save(layout_generator.state_dict(), os.path.join(args.save_dir, f"layout_epoch_{epoch+1}.pt"))
        _write_training_stats(
            args.save_dir,
            epoch + 1,
            semantic_mean=avg_semantic,
            distinctive_mean=avg_distinctive,
            siamese_mean=avg_siamese,
            layout_mean=avg_layout,
        )
        print(
            f"Epoch {epoch+1} complete | "
            f"avg_semantic={avg_semantic:.4f} "
            f"avg_distinctive={avg_distinctive:.4f} "
            f"avg_siamese={avg_siamese:.4f} "
            f"avg_layout={avg_layout:.4f}"
        )


if __name__ == "__main__":
    main()
