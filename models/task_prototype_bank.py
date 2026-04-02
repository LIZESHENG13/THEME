import torch
from torch import nn


class TaskPrototypeBank(nn.Module):
    # TE-DMoE: lightweight EMA prototype anchors for hierarchical region evidence stabilization.
    def __init__(self, num_tasks=3, evidence_dim_per_task=64, momentum=0.95):
        super().__init__()
        self.num_tasks = num_tasks
        self.evidence_dim_per_task = evidence_dim_per_task
        self.momentum = float(momentum)
        self.register_buffer('prototypes', torch.zeros(num_tasks, evidence_dim_per_task))
        self.register_buffer('initialized', torch.zeros(num_tasks, dtype=torch.bool))

    def get(self):
        return self.prototypes

    @torch.no_grad()
    def update(self, task_evidence, modality_available_mask=None):
        # task_evidence: [B, M, T, D]
        if task_evidence.numel() == 0:
            return

        if modality_available_mask is None:
            valid_weight = torch.ones(
                task_evidence.shape[0], task_evidence.shape[1],
                device=task_evidence.device, dtype=task_evidence.dtype
            )
        else:
            valid_weight = modality_available_mask.float()

        for task_idx in range(self.num_tasks):
            task_feat = task_evidence[:, :, task_idx, :]  # [B, M, D]
            weight = valid_weight.unsqueeze(-1)  # [B, M, 1]
            denom = weight.sum().clamp_min(1.0)
            cur_mean = (task_feat * weight).sum(dim=(0, 1)) / denom

            if (denom <= 1.0) and (not bool(self.initialized[task_idx])):
                continue

            if not bool(self.initialized[task_idx]):
                self.prototypes[task_idx] = cur_mean
                self.initialized[task_idx] = True
            else:
                self.prototypes[task_idx] = self.momentum * self.prototypes[task_idx] + \
                    (1.0 - self.momentum) * cur_mean
