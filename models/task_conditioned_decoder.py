import torch
from torch import nn


class TaskConditionedDecoder(nn.Module):
    # TE-DMoE: evidence-guided decoder using region-wise FiLM modulation over fused spatial features.
    def __init__(
            self,
            feature_dim,
            evidence_dim_per_task=64,
            num_tasks=3,
            hidden_dim=128,
            use_hierarchical_output_composition=True,
            hierarchical_region_names=("R1_ET", "R2_TC_MINUS_ET", "R3_WT_MINUS_TC")
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.evidence_dim_per_task = evidence_dim_per_task
        self.num_tasks = num_tasks
        self.use_hierarchical_output_composition = use_hierarchical_output_composition
        self.hierarchical_region_names = tuple(hierarchical_region_names)

        self.gamma_head_ls = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(evidence_dim_per_task),
                nn.Linear(evidence_dim_per_task, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, feature_dim)
            )
            for _ in range(num_tasks)
        ])
        self.beta_head_ls = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(evidence_dim_per_task),
                nn.Linear(evidence_dim_per_task, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, feature_dim)
            )
            for _ in range(num_tasks)
        ])
        self.task_seg_head_ls = nn.ModuleList([
            nn.Conv3d(in_channels=feature_dim, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
            for _ in range(num_tasks)
        ])

    @staticmethod
    def _split_task_evidence(completed_task_evidence, num_tasks):
        # completed_task_evidence: [B, M, E_total]
        total_dim = completed_task_evidence.shape[-1]
        if total_dim % num_tasks != 0:
            raise ValueError(f'Completed evidence dim {total_dim} is not divisible by num_tasks={num_tasks}.')
        dim_per_task = total_dim // num_tasks
        return completed_task_evidence.view(
            completed_task_evidence.shape[0], completed_task_evidence.shape[1], num_tasks, dim_per_task
        )

    # TE-DMoE: compose final BraTS logits [WT, TC, ET] from hierarchical region logits.
    @staticmethod
    def _compose_hierarchical_logits(logits_r1, logits_r2, logits_r3):
        # TE-DMoE: R1=ET, R2=TC-ET, R3=WT-TC.
        logits_et = logits_r1
        logits_tc = logits_r1 + logits_r2
        logits_wt = logits_r1 + logits_r2 + logits_r3
        return torch.cat([logits_wt, logits_tc, logits_et], dim=1)

    def forward(self, fused_feature, completed_task_evidence):
        # fused_feature: [B, C, H, W, D], completed_task_evidence: [B, M, E_total]
        split_evidence = self._split_task_evidence(completed_task_evidence, self.num_tasks)  # [B, M, T, D]
        task_summary = split_evidence.mean(dim=1)  # [B, T, D]

        gamma_r1 = self.gamma_head_ls[0](task_summary[:, 0, :]).view(-1, self.feature_dim, 1, 1, 1)
        gamma_r2 = self.gamma_head_ls[1](task_summary[:, 1, :]).view(-1, self.feature_dim, 1, 1, 1)
        gamma_r3 = self.gamma_head_ls[2](task_summary[:, 2, :]).view(-1, self.feature_dim, 1, 1, 1)
        beta_r1 = self.beta_head_ls[0](task_summary[:, 0, :]).view(-1, self.feature_dim, 1, 1, 1)
        beta_r2 = self.beta_head_ls[1](task_summary[:, 1, :]).view(-1, self.feature_dim, 1, 1, 1)
        beta_r3 = self.beta_head_ls[2](task_summary[:, 2, :]).view(-1, self.feature_dim, 1, 1, 1)

        logits_r1 = self.task_seg_head_ls[0](fused_feature * (1.0 + gamma_r1) + beta_r1)
        logits_r2 = self.task_seg_head_ls[1](fused_feature * (1.0 + gamma_r2) + beta_r2)
        logits_r3 = self.task_seg_head_ls[2](fused_feature * (1.0 + gamma_r3) + beta_r3)

        if self.use_hierarchical_output_composition:
            logits = self._compose_hierarchical_logits(logits_r1, logits_r2, logits_r3)
        else:
            # TE-DMoE: legacy direct-mode fallback; branch order is [ET, TC, WT].
            logits = torch.cat([logits_r3, logits_r2, logits_r1], dim=1)

        return logits, {
            'task_summary': task_summary,
            'region_summary': task_summary,
            'region_logits': torch.cat([logits_r1, logits_r2, logits_r3], dim=1),
            'logits_r1': logits_r1,
            'logits_r2': logits_r2,
            'logits_r3': logits_r3,
            'gamma_r1': gamma_r1,
            'gamma_r2': gamma_r2,
            'gamma_r3': gamma_r3,
            'beta_r1': beta_r1,
            'beta_r2': beta_r2,
            'beta_r3': beta_r3,
            'hierarchical_region_names': self.hierarchical_region_names,
            'use_hierarchical_output_composition': self.use_hierarchical_output_composition,
        }

    def _decode_single_task(self, fused_feature, task_evidence, task_idx):
        gamma = self.gamma_head_ls[task_idx](task_evidence).view(-1, self.feature_dim, 1, 1, 1)
        beta = self.beta_head_ls[task_idx](task_evidence).view(-1, self.feature_dim, 1, 1, 1)
        modulated = fused_feature * (1.0 + gamma) + beta
        return self.task_seg_head_ls[task_idx](modulated)
