import torch
from torch import nn


class TaskSpatialEvidenceFusion(nn.Module):
    # TE-DMoE: task-aware spatial gating branch with hierarchical region-logit composition.
    def __init__(
            self,
            feature_dim,
            evidence_dim_per_task,
            num_tasks=3,
            hidden_dim=64,
            zero_init=True,
            alpha_init=0.0,
            gate_bias_init=-2.0,
            hierarchical_region_names=("R1_ET", "R2_TC_MINUS_ET", "R3_WT_MINUS_TC")
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.evidence_dim_per_task = evidence_dim_per_task
        self.num_tasks = num_tasks
        self.hierarchical_region_names = tuple(hierarchical_region_names)
        self.gate_bias_init = float(gate_bias_init)
        if self.num_tasks != 3:
            raise ValueError(f'TaskSpatialEvidenceFusion expects num_tasks=3, got {self.num_tasks}.')

        self.task_evidence_proj = nn.ModuleList([
            nn.Linear(evidence_dim_per_task, hidden_dim)
            for _ in range(self.num_tasks)
        ])
        self.task_gate_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(feature_dim + hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_dim, 1, kernel_size=3, stride=1, padding=1, bias=True),
            )
            for _ in range(self.num_tasks)
        ])
        self.task_transform = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(feature_dim, feature_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.InstanceNorm3d(feature_dim, eps=1e-5, affine=True),
                nn.LeakyReLU(inplace=True),
            )
            for _ in range(self.num_tasks)
        ])
        self.task_region_head = nn.ModuleList([
            nn.Conv3d(feature_dim, 1, kernel_size=1, stride=1, padding=0, bias=True)
            for _ in range(self.num_tasks)
        ])
        self.task_alpha = nn.Parameter(torch.full((self.num_tasks,), float(alpha_init), dtype=torch.float32))

        if zero_init:
            for gate_head in self.task_gate_head:
                nn.init.zeros_(gate_head[-1].weight)
                # TE-DMoE: initialize gate logits to a negative bias so gates start near-closed.
                nn.init.constant_(gate_head[-1].bias, self.gate_bias_init)

    @staticmethod
    def compose_hierarchical_logits(logits_r1, logits_r2, logits_r3):
        # TE-DMoE: R1=ET, R2=TC-ET, R3=WT-TC -> final [WT, TC, ET].
        logits_et = logits_r1
        logits_tc = logits_r1 + logits_r2
        logits_wt = logits_r1 + logits_r2 + logits_r3
        return torch.cat([logits_wt, logits_tc, logits_et], dim=1)

    def forward(self, fused_feature, task_evidence_summary):
        # fused_feature: [B, C, H, W, D], task_evidence_summary: [B, T, D_e]
        if task_evidence_summary.shape[1] != self.num_tasks:
            raise ValueError(
                f'task_evidence_summary.shape[1]={task_evidence_summary.shape[1]} '
                f'does not match num_tasks={self.num_tasks}.'
            )

        spatial_shape = fused_feature.shape[2:]
        gated_feature = fused_feature
        task_gate_map_ls = []
        task_delta_map_ls = []
        task_region_logits_ls = []

        for task_idx in range(self.num_tasks):
            task_evidence = task_evidence_summary[:, task_idx, :]  # [B, D_e]
            task_context = self.task_evidence_proj[task_idx](task_evidence)  # [B, H]
            task_context_map = task_context.view(task_context.shape[0], task_context.shape[1], 1, 1, 1).expand(
                -1, -1, spatial_shape[0], spatial_shape[1], spatial_shape[2]
            )

            gate_input = torch.cat([fused_feature, task_context_map], dim=1)
            task_gate_map = torch.sigmoid(self.task_gate_head[task_idx](gate_input))  # [B,1,H,W,D]
            task_delta = self.task_transform[task_idx](fused_feature)
            alpha = self.task_alpha[task_idx].view(1, 1, 1, 1, 1)
            task_modulated_feature = fused_feature + alpha * task_gate_map * task_delta
            task_region_logits = self.task_region_head[task_idx](task_modulated_feature)  # [B,1,H,W,D]

            gated_feature = gated_feature + alpha * task_gate_map * task_delta
            task_gate_map_ls.append(task_gate_map)
            # TE-DMoE: keep lightweight delta maps for diagnostics without storing full C-channel tensors.
            task_delta_map_ls.append(torch.norm(task_delta, dim=1, keepdim=True))
            task_region_logits_ls.append(task_region_logits)

        logits_r1, logits_r2, logits_r3 = task_region_logits_ls
        logits = self.compose_hierarchical_logits(logits_r1, logits_r2, logits_r3)
        task_gate_maps = torch.cat(task_gate_map_ls, dim=1)  # [B, 3, H, W, D]
        delta_r1, delta_r2, delta_r3 = task_delta_map_ls
        gate_r1, gate_r2, gate_r3 = task_gate_map_ls
        return logits, gated_feature, {
            'task_gate_maps': task_gate_maps,
            'gate_r1': gate_r1,
            'gate_r2': gate_r2,
            'gate_r3': gate_r3,
            'delta_r1': delta_r1,
            'delta_r2': delta_r2,
            'delta_r3': delta_r3,
            'alpha_r1': self.task_alpha[0],
            'alpha_r2': self.task_alpha[1],
            'alpha_r3': self.task_alpha[2],
            'region_logits': torch.cat(task_region_logits_ls, dim=1),  # [B, 3, H, W, D]
            'logits_r1': logits_r1,
            'logits_r2': logits_r2,
            'logits_r3': logits_r3,
            'task_alpha': self.task_alpha,
            'hierarchical_region_names': self.hierarchical_region_names,
        }
