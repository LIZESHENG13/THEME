import torch
from torch import nn


class MissingEvidencePredictor(nn.Module):
    # TE-DMoE: hierarchical missing-evidence completion with optional prototype-conditioned cross-modal attention.
    def __init__(
            self,
            evidence_dim,
            num_modalities=4,
            hidden_dim=256,
            predictor_type='prototype_cross_attn',
            use_shared_predictor=True,
            use_learnable_scale=True,
            init_scale=1.0,
            num_tasks=3,
            use_task_prototype_query=True
    ):
        super().__init__()
        if predictor_type not in ['mlp', 'prototype_cross_attn']:
            raise ValueError(f'Unsupported missing evidence predictor type: {predictor_type}')

        self.evidence_dim = evidence_dim
        self.num_modalities = num_modalities
        self.predictor_type = predictor_type
        self.use_shared_predictor = use_shared_predictor
        self.num_tasks = num_tasks
        if self.evidence_dim % self.num_tasks != 0:
            raise ValueError(f'evidence_dim={self.evidence_dim} must be divisible by {self.num_tasks}.')
        self.evidence_dim_per_task = self.evidence_dim // self.num_tasks
        self.use_learnable_scale = use_learnable_scale
        self.use_task_prototype_query = use_task_prototype_query

        self.modality_embedding = nn.Embedding(num_modalities, self.evidence_dim_per_task)
        if self.use_task_prototype_query:
            self.task_prototypes = nn.Parameter(
                torch.randn(self.num_tasks, self.evidence_dim_per_task) * 0.02
            )
        else:
            self.register_parameter('task_prototypes', None)

        if self.predictor_type == 'mlp':
            self._build_mlp_predictor(hidden_dim)
        else:
            self._build_prototype_cross_attn_predictor(hidden_dim)

        if self.use_learnable_scale:
            self.region_output_scale = nn.Parameter(
                torch.full((self.num_tasks,), float(init_scale), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'region_output_scale',
                torch.full((self.num_tasks,), float(init_scale), dtype=torch.float32)
            )

    def _build_mlp_predictor(self, hidden_dim):
        summary_dim = 3 * self.evidence_dim_per_task
        in_dim_r1 = summary_dim + self.evidence_dim_per_task
        in_dim_r2 = summary_dim + self.evidence_dim_per_task + self.evidence_dim_per_task
        in_dim_r3 = summary_dim + self.evidence_dim_per_task + 2 * self.evidence_dim_per_task
        self.region_predictor_r1 = nn.Sequential(
            nn.Linear(in_dim_r1, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, self.evidence_dim_per_task),
        )
        self.region_predictor_r2 = nn.Sequential(
            nn.Linear(in_dim_r2, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, self.evidence_dim_per_task),
        )
        self.region_predictor_r3 = nn.Sequential(
            nn.Linear(in_dim_r3, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, self.evidence_dim_per_task),
        )

        self.modality_region_predictor_r1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim_r1, hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(hidden_dim, self.evidence_dim_per_task),
            )
            for _ in range(self.num_modalities)
        ])
        self.modality_region_predictor_r2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim_r2, hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(hidden_dim, self.evidence_dim_per_task),
            )
            for _ in range(self.num_modalities)
        ])
        self.modality_region_predictor_r3 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim_r3, hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(hidden_dim, self.evidence_dim_per_task),
            )
            for _ in range(self.num_modalities)
        ])

    def _build_prototype_cross_attn_predictor(self, hidden_dim):
        d = self.evidence_dim_per_task
        # TE-DMoE: query from [target modality embedding; task prototype].
        self.task_query_proj_ls = nn.ModuleList([
            nn.Linear(2 * d, d)
            for _ in range(self.num_tasks)
        ])
        # TE-DMoE: key/value from [observed task evidence; modality embedding; reliability].
        self.task_key_proj_ls = nn.ModuleList([
            nn.Linear(2 * d + 1, d)
            for _ in range(self.num_tasks)
        ])
        self.task_value_proj_ls = nn.ModuleList([
            nn.Linear(2 * d + 1, d)
            for _ in range(self.num_tasks)
        ])
        self.task_context_proj_ls = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, d),
            )
            for _ in range(self.num_tasks)
        ])

        # TE-DMoE: hierarchical residual heads R1 -> R2 -> R3.
        self.cross_attn_region_head_r1 = nn.Sequential(
            nn.Linear(3 * d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d),
        )
        self.cross_attn_region_head_r2 = nn.Sequential(
            nn.Linear(4 * d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d),
        )
        self.cross_attn_region_head_r3 = nn.Sequential(
            nn.Linear(5 * d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d),
        )

    # TE-DMoE: reshape [B, M, E] -> [B, M, T, D].
    def _split_task_evidence(self, evidence):
        return evidence.view(
            evidence.shape[0], evidence.shape[1], self.num_tasks, self.evidence_dim_per_task
        )

    # TE-DMoE: compute region-wise summary from available modalities only.
    def _compute_task_summary(self, observed_task_evidence, modality_available_mask):
        # observed_task_evidence: [B, M, D], modality_available_mask: [B, M]
        available_mask = modality_available_mask.float().unsqueeze(-1)
        available_count = modality_available_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        summary = (observed_task_evidence * available_mask).sum(dim=1) / available_count
        all_missing_mask = modality_available_mask.sum(dim=1, keepdim=True) == 0
        summary = torch.where(all_missing_mask, torch.zeros_like(summary), summary)
        return summary

    @staticmethod
    def _masked_softmax(scores, source_available_mask, eps=1e-8):
        # scores: [B, M_target, M_source], source_available_mask: [B, M_source]
        source_mask = source_available_mask.unsqueeze(1)  # [B,1,M_source]
        masked_scores = scores.masked_fill(~source_mask, -1e4)
        weights = torch.softmax(masked_scores, dim=-1)
        weights = weights * source_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
        return weights

    def _predict_with_mlp(self, split_observed, modality_available_mask):
        batch_size = split_observed.shape[0]
        modality_ids = torch.arange(self.num_modalities, device=split_observed.device)
        modality_embed = self.modality_embedding(modality_ids).unsqueeze(0).expand(batch_size, -1, -1)  # [B, M, D]
        summary_r1 = self._compute_task_summary(split_observed[:, :, 0, :], modality_available_mask)  # [B, D]
        summary_r2 = self._compute_task_summary(split_observed[:, :, 1, :], modality_available_mask)  # [B, D]
        summary_r3 = self._compute_task_summary(split_observed[:, :, 2, :], modality_available_mask)  # [B, D]
        summary_all = torch.cat([summary_r1, summary_r2, summary_r3], dim=-1)  # [B, 3D]
        summary_expand = summary_all.unsqueeze(1).expand(-1, self.num_modalities, -1)  # [B, M, 3D]
        base_task = summary_expand.view(batch_size, self.num_modalities, self.num_tasks, self.evidence_dim_per_task)

        predictor_input_r1 = torch.cat([summary_expand, modality_embed], dim=-1)
        if self.use_shared_predictor:
            delta_r1 = self.region_predictor_r1(predictor_input_r1)
        else:
            delta_r1 = torch.stack([
                self.modality_region_predictor_r1[m](predictor_input_r1[:, m, :])
                for m in range(self.num_modalities)
            ], dim=1)
        pred_r1 = base_task[:, :, 0, :] + delta_r1

        predictor_input_r2 = torch.cat([summary_expand, modality_embed, pred_r1], dim=-1)
        if self.use_shared_predictor:
            delta_r2 = self.region_predictor_r2(predictor_input_r2)
        else:
            delta_r2 = torch.stack([
                self.modality_region_predictor_r2[m](predictor_input_r2[:, m, :])
                for m in range(self.num_modalities)
            ], dim=1)
        pred_r2 = base_task[:, :, 1, :] + delta_r2

        predictor_input_r3 = torch.cat([summary_expand, modality_embed, pred_r1, pred_r2], dim=-1)
        if self.use_shared_predictor:
            delta_r3 = self.region_predictor_r3(predictor_input_r3)
        else:
            delta_r3 = torch.stack([
                self.modality_region_predictor_r3[m](predictor_input_r3[:, m, :])
                for m in range(self.num_modalities)
            ], dim=1)
        base_task = base_task
        delta_task = torch.stack([delta_r1, delta_r2, delta_r3], dim=2)  # [B, M, T, D]
        return base_task, delta_task

    def _predict_with_prototype_cross_attn(self, split_observed, modality_available_mask, modality_reliability):
        # split_observed: [B, M, T, D]
        bsz, _, _, d = split_observed.shape
        modality_ids = torch.arange(self.num_modalities, device=split_observed.device)
        modality_embed = self.modality_embedding(modality_ids).unsqueeze(0).expand(bsz, -1, -1)  # [B, M, D]
        if modality_reliability is None:
            modality_reliability = modality_available_mask.float()
        reliability = modality_reliability.unsqueeze(-1)  # [B, M, 1]

        base_task_ls = []
        context_task_ls = []
        for task_idx in range(self.num_tasks):
            summary_t = self._compute_task_summary(split_observed[:, :, task_idx, :], modality_available_mask)  # [B, D]
            base_task_ls.append(summary_t.unsqueeze(1).expand(-1, self.num_modalities, -1))  # [B, M, D]

            if self.use_task_prototype_query and self.task_prototypes is not None:
                proto_t = self.task_prototypes[task_idx].view(1, 1, d).expand(bsz, self.num_modalities, -1)  # [B,M,D]
            else:
                proto_t = torch.zeros_like(modality_embed)
            query_input = torch.cat([modality_embed, proto_t], dim=-1)  # [B, M, 2D]
            query = self.task_query_proj_ls[task_idx](query_input)      # [B, M, D]

            kv_input = torch.cat([split_observed[:, :, task_idx, :], modality_embed, reliability], dim=-1)  # [B,M,2D+1]
            key = self.task_key_proj_ls[task_idx](kv_input)             # [B, M, D]
            value = self.task_value_proj_ls[task_idx](kv_input)         # [B, M, D]

            score = torch.matmul(query, key.transpose(1, 2)) / (float(d) ** 0.5)  # [B, M_target, M_source]
            attn = self._masked_softmax(score, modality_available_mask)
            context = torch.matmul(attn, value)  # [B, M_target, D]
            context_task_ls.append(self.task_context_proj_ls[task_idx](context))

        base_task = torch.stack(base_task_ls, dim=2)      # [B, M, T, D]
        context_task = torch.stack(context_task_ls, dim=2)  # [B, M, T, D]

        base_r1 = base_task[:, :, 0, :]
        base_r2 = base_task[:, :, 1, :]
        base_r3 = base_task[:, :, 2, :]
        context_r1 = context_task[:, :, 0, :]
        context_r2 = context_task[:, :, 1, :]
        context_r3 = context_task[:, :, 2, :]

        delta_r1 = self.cross_attn_region_head_r1(torch.cat([context_r1, base_r1, modality_embed], dim=-1))
        pred_r1 = base_r1 + delta_r1
        delta_r2 = self.cross_attn_region_head_r2(torch.cat([context_r2, base_r2, modality_embed, pred_r1], dim=-1))
        pred_r2 = base_r2 + delta_r2
        delta_r3 = self.cross_attn_region_head_r3(
            torch.cat([context_r3, base_r3, modality_embed, pred_r1, pred_r2], dim=-1)
        )
        delta_task = torch.stack([delta_r1, delta_r2, delta_r3], dim=2)  # [B, M, T, D]
        return base_task, delta_task

    def forward(self, observed_evidence, modality_available_mask, modality_reliability=None, return_summary=False):
        # observed_evidence: [B, M, E_total], modality_available_mask: [B, M]
        split_observed = self._split_task_evidence(observed_evidence)  # [B, M, T, D]
        if self.predictor_type == 'mlp':
            base_task, delta_task = self._predict_with_mlp(split_observed, modality_available_mask)
        else:
            base_task, delta_task = self._predict_with_prototype_cross_attn(
                split_observed, modality_available_mask, modality_reliability
            )

        scale = self.region_output_scale.view(1, 1, self.num_tasks, 1).to(delta_task.device)
        predicted_task = base_task + scale * delta_task  # z_pred = z_bar + delta_z
        predicted_flat = predicted_task.reshape(observed_evidence.shape[0], self.num_modalities, self.evidence_dim)

        if return_summary:
            base_flat = base_task.reshape(observed_evidence.shape[0], self.num_modalities, self.evidence_dim)
            return predicted_flat, base_flat
        return predicted_flat

