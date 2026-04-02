import torch
from torch import nn


class EvidenceFiLMBottleneck(nn.Module):
    # TE-DMoE: strong evidence coupling via bottleneck FiLM with identity-friendly residual scaling.
    def __init__(
            self,
            feature_dim,
            evidence_dim,
            hidden_dim=128,
            use_spatial_attention=True,
            spatial_attention_channels=64,
            zero_init=True,
            alpha_init=0.0
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.evidence_dim = evidence_dim
        self.use_spatial_attention = use_spatial_attention
        self.spatial_attention_channels = spatial_attention_channels

        self.film_gamma_mlp = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.film_beta_mlp = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # TE-DMoE: learnable residual scales, zero-init keeps startup close to baseline.
        self.alpha_gamma = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_beta = nn.Parameter(torch.tensor(float(alpha_init)))

        if zero_init:
            nn.init.zeros_(self.film_gamma_mlp[-1].weight)
            nn.init.zeros_(self.film_gamma_mlp[-1].bias)
            nn.init.zeros_(self.film_beta_mlp[-1].weight)
            nn.init.zeros_(self.film_beta_mlp[-1].bias)

        if self.use_spatial_attention:
            self.spatial_proj = nn.Linear(evidence_dim, spatial_attention_channels)
            self.spatial_attention_head = nn.Sequential(
                nn.Conv3d(feature_dim + spatial_attention_channels, spatial_attention_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(spatial_attention_channels, 1, kernel_size=3, padding=1),
            )
            if zero_init:
                nn.init.zeros_(self.spatial_attention_head[-1].weight)
                nn.init.zeros_(self.spatial_attention_head[-1].bias)

    def forward(self, fused_feature, evidence_condition_vector):
        # fused_feature: [B, C, H, W, D], evidence_condition_vector: [B, E]
        gamma_raw = self.film_gamma_mlp(evidence_condition_vector)  # [B, C]
        beta_raw = self.film_beta_mlp(evidence_condition_vector)  # [B, C]

        gamma = 1.0 + self.alpha_gamma * gamma_raw
        beta = self.alpha_beta * beta_raw

        gamma_map = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1, 1)
        beta_map = beta.view(beta.shape[0], beta.shape[1], 1, 1, 1)
        film_feature = gamma_map * fused_feature + beta_map

        if self.use_spatial_attention:
            z_proj = self.spatial_proj(evidence_condition_vector)  # [B, C_att]
            z_map = z_proj.view(
                z_proj.shape[0], z_proj.shape[1], 1, 1, 1
            ).expand(-1, -1, fused_feature.shape[2], fused_feature.shape[3], fused_feature.shape[4])
            spatial_input = torch.cat([fused_feature, z_map], dim=1)
            spatial_attention = torch.sigmoid(self.spatial_attention_head(spatial_input))  # [B,1,H,W,D]
        else:
            spatial_attention = fused_feature.new_ones(
                (fused_feature.shape[0], 1, fused_feature.shape[2], fused_feature.shape[3], fused_feature.shape[4])
            )

        modulated_feature = spatial_attention * film_feature + (1.0 - spatial_attention) * fused_feature
        return modulated_feature, {
            'film_gamma': gamma,
            'film_beta': beta,
            'film_gamma_raw': gamma_raw,
            'film_beta_raw': beta_raw,
            'film_alpha_gamma': self.alpha_gamma,
            'film_alpha_beta': self.alpha_beta,
            'bottleneck_spatial_attention': spatial_attention,
            'fused_feature_before_film': fused_feature,
            'fused_feature_after_film': modulated_feature,
        }

