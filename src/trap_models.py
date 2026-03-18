import torch
import torch.nn as nn


class SiameseSemanticNetwork(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 512,
        text_embed_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 512,
    ):
        super().__init__()
        self.common_branch = nn.Sequential(
            nn.Linear(image_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self.distinctive_branch = nn.Sequential(
            nn.Linear(image_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self.text_projection = nn.Linear(text_embed_dim, output_dim)
        self.semantic_bn = nn.BatchNorm1d(output_dim, affine=True)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, image_embed, text_embed=None, mode: str = "both"):
        common_features = self.common_branch(image_embed)
        distinctive_features = self.distinctive_branch(image_embed)

        if mode == "common":
            return common_features
        if mode == "distinctive":
            return distinctive_features

        if text_embed is not None:
            projected_text = self.text_projection(text_embed)
            distinctive_features = self.semantic_bn(distinctive_features)
            semantic_weight = torch.sigmoid(projected_text)
            modulated_distinctive = distinctive_features * semantic_weight
            return common_features + modulated_distinctive, common_features, modulated_distinctive

        return common_features + distinctive_features, common_features, distinctive_features


class SemanticLayoutGenerator(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 512,
        text_embed_dim: int = 512,
        expanded_text_dim: int = 1024,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.text_expand = nn.Linear(text_embed_dim, expanded_text_dim)
        input_dim = image_embed_dim + expanded_text_dim  # 512 + 1024 = 1536 (paper setting)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, text_embed, image_embed):
        text_expanded = self.text_expand(text_embed)
        combined = torch.cat([text_expanded, image_embed], dim=1)
        encoded = self.encoder(combined)
        batch_size = encoded.shape[0]
        reshaped = encoded.view(batch_size, 256, 2, 2)
        return self.decoder(reshaped)


class ClipToPooledMapper(nn.Module):
    """Maps CLIP-latent features into SD pooled-prompt space."""

    def __init__(self, clip_dim: int = 512, hidden_dim: int = 1024, pooled_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pooled_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        return self.net(clip_features)
