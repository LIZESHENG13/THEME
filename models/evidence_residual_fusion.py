import torch
from torch import nn


class EvidenceResidualFusion(nn.Module):
    # TE-DMoE: lightweight residual branch for evidence-conditioned task-wise logit modulation.
    def __init__(
            self,
            evidence_dim,
            num_classes=3,
            hidden_dim=128,
            residual_type='bias',
            residual_scale=0.1,
            learnable_scale=False,
            gamma_tanh=True
    ):
        super().__init__()
        if residual_type not in ['bias', 'affine']:
            raise ValueError(f'Unsupported evidence residual type: {residual_type}')

        self.evidence_dim = evidence_dim
        self.num_classes = num_classes
        self.residual_type = residual_type
        self.residual_scale = float(residual_scale)
        self.learnable_scale = learnable_scale
        self.gamma_tanh = gamma_tanh

        # TE-DMoE: beta branch (bias-like correction) kept on original name for checkpoint compatibility.
        self.residual_mlp = nn.Sequential(
            nn.Linear(evidence_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )
        # TE-DMoE: affine mode adds a small gamma branch for multiplicative task-wise modulation.
        self.gamma_mlp = None
        if self.residual_type == 'affine':
            self.gamma_mlp = nn.Sequential(
                nn.Linear(evidence_dim, hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(hidden_dim, num_classes),
            )

        # TE-DMoE: optional learnable scalar kept stable by zero initialization and tanh bounding.
        if self.learnable_scale:
            self.learnable_scale_param = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('learnable_scale_param', None)
        # TE-DMoE: runtime multiplier enables warmup/ramp schedules without changing checkpoint structure.
        self.register_buffer('runtime_scale_multiplier', torch.tensor(1.0))

    # TE-DMoE: set dynamic residual scaling (e.g., curriculum warmup) from trainer.
    def set_runtime_scale_multiplier(self, multiplier):
        if not torch.is_tensor(multiplier):
            multiplier = torch.tensor(float(multiplier), device=self.runtime_scale_multiplier.device)
        self.runtime_scale_multiplier.copy_(multiplier.to(self.runtime_scale_multiplier.device))

    def _get_residual_scale(self, reference_tensor):
        if self.learnable_scale_param is None:
            base_scale = reference_tensor.new_tensor(self.residual_scale)
        else:
            bounded_scale = torch.tanh(self.learnable_scale_param).to(dtype=reference_tensor.dtype)
            base_scale = reference_tensor.new_tensor(self.residual_scale) * bounded_scale

        runtime_mul = self.runtime_scale_multiplier.to(
            device=reference_tensor.device, dtype=reference_tensor.dtype
        )
        return base_scale * runtime_mul

    def _predict_gamma_beta(self, evidence_summary):
        beta = self.residual_mlp(evidence_summary).view(-1, self.num_classes, 1, 1, 1)
        if self.residual_type == 'bias':
            gamma = torch.zeros_like(beta)
            return gamma, beta

        gamma = self.gamma_mlp(evidence_summary).view(-1, self.num_classes, 1, 1, 1)
        if self.gamma_tanh:
            gamma = torch.tanh(gamma)
        return gamma, beta

    def forward(self, original_logits, residual_task_evidence):
        if residual_task_evidence is None:
            raise ValueError('residual_task_evidence is required for evidence residual fusion.')

        evidence_summary = residual_task_evidence.mean(dim=1)
        gamma, beta = self._predict_gamma_beta(evidence_summary)
        residual_scale = self._get_residual_scale(original_logits)
        evidence_residual_logits = (
            original_logits * gamma + beta
        ).expand_as(original_logits)
        final_logits = original_logits + residual_scale.view(1, 1, 1, 1, 1) * evidence_residual_logits

        return final_logits, evidence_residual_logits, gamma, beta, residual_scale
