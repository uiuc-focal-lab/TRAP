import argparse
import os
from pathlib import Path

import filelock
from filelock import SoftFileLock
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Pre-cache HF VLM models into HF_HOME cache.")
    parser.add_argument("--repo_id", type=str, default="llava-hf/llava-v1.6-34b-hf")
    parser.add_argument(
        "--repo_ids",
        type=str,
        default="",
        help="Optional comma-separated repo ids. If set, overrides --repo_id.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    filelock.FileLock = SoftFileLock

    repos: list[str] = []
    if args.repo_ids.strip():
        repos = [r.strip() for r in args.repo_ids.split(",") if r.strip()]
    if not repos:
        repos = [args.repo_id]
    repos = list(dict.fromkeys(repos))

    print(f"HF_HOME={os.environ.get('HF_HOME')}")
    print(f"Downloading {len(repos)} repo(s): {repos}")

    for repo_id in repos:
        print(f"Downloading snapshot: {repo_id} (revision={args.revision})")
        snap_dir = snapshot_download(
            repo_id=repo_id,
            revision=args.revision,
            local_files_only=False,
            max_workers=args.max_workers,
            tqdm_class=None,
        )

        # Many VLM repos reference a separate vision tower. Cache it too so eval can run with local_files_only.
        try:
            cfg_path = Path(snap_dir) / "config.json"
            if cfg_path.exists():
                import json

                cfg = json.loads(cfg_path.read_text())
                vision_tower = cfg.get("mm_vision_tower") or cfg.get("vision_tower")
                if isinstance(vision_tower, str) and "/" in vision_tower:
                    print(f"Also downloading vision tower: {vision_tower}")
                    snapshot_download(
                        repo_id=vision_tower,
                        revision=None,
                        local_files_only=False,
                        max_workers=args.max_workers,
                        tqdm_class=None,
                    )
        except Exception as e:
            print(f"[WARN] Failed to precache vision tower for {repo_id}: {e!r}")
    print("Done.")


if __name__ == "__main__":
    main()
