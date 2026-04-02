import torch
from torch import nn

from configs_joint_training import ModelConfig
from models.evidence_film import EvidenceFiLMBottleneck
from models.evidence_residual_fusion import EvidenceResidualFusion
from models.missing_evidence_predictor import MissingEvidencePredictor
from models.nnunet import UNet
from models.task_spatial_evidence_fusion import TaskSpatialEvidenceFusion
from models.task_conditioned_decoder import TaskConditionedDecoder
from models.task_prototype_bank import TaskPrototypeBank


# TE-DMoE: partially load checkpoints by key+shape match and print compatibility report.
def _load_partial_state_dict_with_report(module, state_dict, module_name='module', skip_prefixes=None):
    if skip_prefixes is None:
        skip_prefixes = []

    module_state = module.state_dict()
    filtered_state = {}
    skipped_by_prefix = []
    skipped_by_shape = []
    unexpected_keys = []

    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped_by_prefix.append(key)
            continue
        if key not in module_state:
            unexpected_keys.append(key)
            continue
        if module_state[key].shape != value.shape:
            skipped_by_shape.append(key)
            continue
        filtered_state[key] = value

    load_res = module.load_state_dict(filtered_state, strict=False)
    missing_keys = list(load_res.missing_keys)
    loaded_count = len(filtered_state)

    print(
        f"[TE-DMoE][partial_load:{module_name}] "
        f"loaded={loaded_count}, missing={len(missing_keys)}, "
        f"unexpected={len(unexpected_keys)}, skip_prefix={len(skipped_by_prefix)}, "
        f"skip_shape={len(skipped_by_shape)}"
    )
    if len(missing_keys) > 0:
        print(f"[TE-DMoE][partial_load:{module_name}] missing_keys(sample)={missing_keys[:8]}")
    if len(unexpected_keys) > 0:
        print(f"[TE-DMoE][partial_load:{module_name}] unexpected_keys(sample)={unexpected_keys[:8]}")
    if len(skipped_by_shape) > 0:
        print(f"[TE-DMoE][partial_load:{module_name}] skipped_shape(sample)={skipped_by_shape[:8]}")
    if len(skipped_by_prefix) > 0:
        print(f"[TE-DMoE][partial_load:{module_name}] skipped_prefix(sample)={skipped_by_prefix[:8]}")


# TE-DMoE: skip evidence-head params when reusing old experts across evidence semantic changes.
def _get_expert_partial_skip_prefixes():
    return [
        'task_score_head.',
        'task_token_proj.',
        'task_token_agg.',
        'evidence_head.',
        'evidence_pool.',
    ]


# TE-DMoE: standardize optional evidence packaging for joint-training consumers.
def _build_task_evidence_aux(
        stacked_evidence,
        modality_mask,
        predicted_task_evidence=None,
        mean_task_evidence=None,
        missing_fusion_mean_weight=0.7,
        missing_fusion_pred_weight=0.3
):
    if predicted_task_evidence is None:
        predicted_task_evidence = torch.zeros_like(stacked_evidence)
    if mean_task_evidence is None:
        mean_task_evidence = torch.zeros_like(stacked_evidence)
    modality_available_mask = ~modality_mask
    # TE-DMoE: weighted completion for missing slots.
    blended_missing_task_evidence = (
        float(missing_fusion_mean_weight) * mean_task_evidence +
        float(missing_fusion_pred_weight) * predicted_task_evidence
    )
    completed_task_evidence = torch.where(
        modality_available_mask.unsqueeze(-1),
        stacked_evidence,
        blended_missing_task_evidence
    )
    return {
        'task_evidence': stacked_evidence,
        'predicted_task_evidence': predicted_task_evidence,
        'mean_task_evidence': mean_task_evidence,
        'blended_missing_task_evidence': blended_missing_task_evidence,
        'completed_task_evidence': completed_task_evidence,
        # TE-DMoE: alias fields to simplify analysis scripts.
        'z_obs': stacked_evidence,
        'z_pred': predicted_task_evidence,
        'z_bar': mean_task_evidence,
        'z_comp': completed_task_evidence,
        'task_evidence_list': [stacked_evidence[:, modality_idx, :] for modality_idx in range(stacked_evidence.shape[1])],
        'modality_mask': modality_mask,
        'modality_available_mask': modality_available_mask,
        'missing_fusion_mean_weight': float(missing_fusion_mean_weight),
        'missing_fusion_pred_weight': float(missing_fusion_pred_weight),
    }


# TE-DMoE: predict missing evidence from observed evidence only.
def _predict_missing_task_evidence(
        missing_evidence_predictor, observed_task_evidence, modality_mask, modality_reliability=None
):
    if missing_evidence_predictor is None:
        zeros = torch.zeros_like(observed_task_evidence)
        return zeros, zeros

    modality_available_mask = ~modality_mask
    try:
        pred = missing_evidence_predictor(
            observed_task_evidence,
            modality_available_mask,
            modality_reliability=modality_reliability,
            return_summary=True
        )
    except TypeError:
        pred = missing_evidence_predictor(
            observed_task_evidence,
            modality_available_mask,
            return_summary=True
        )
    if isinstance(pred, tuple):
        return pred[0], pred[1]
    # TE-DMoE: fallback for older predictor implementation.
    zeros = torch.zeros_like(observed_task_evidence)
    return pred, zeros


# TE-DMoE: apply optional lightweight residual correction from completed or observed evidence.
def _apply_task_evidence_residual(
        evidence_residual_fusion,
        original_logits,
        evidence_aux,
        use_completed_evidence_for_residual=True
):
    if evidence_residual_fusion is None:
        return original_logits, None, None, None, None
    if evidence_aux is None:
        return original_logits, None, None, None, None

    residual_source_key = 'completed_task_evidence' if use_completed_evidence_for_residual else 'task_evidence'
    residual_task_evidence = evidence_aux.get(residual_source_key, None)
    if residual_task_evidence is None:
        return original_logits, None, None, None, None

    return evidence_residual_fusion(original_logits, residual_task_evidence)


# Implementation for our default DMoME, where the fusion of the experts is operated at the output level (use segmentation map before sigmoid)
class DMoMEOutputLevel(nn.Module):
    def __init__(self, ignore_assert=False):
        super().__init__()
        if ignore_assert:
            print("CAREFUL!!!!!!YOU ARE IGNORING THE ASSERTATION OF MODEL CREATION!!!!!!")
        else:
            assert ModelConfig.TRAIN_LOSS_ARGS['need_sigmoid'] and ModelConfig.VAL_LOSS_ARGS['need_sigmoid'], \
                "loss for this model needs sigmoid!"

        self.pretrained_expert_file_list = ModelConfig.PRETRAINED_EXPERT_FILE_LIST
        self.n_stages = ModelConfig.N_STAGES
        # TE-DMoE: control whether joint model collects expert evidence tensors.
        self.use_task_evidence = getattr(ModelConfig, 'USE_TASK_EVIDENCE', False)
        self.num_evidence_tasks = getattr(ModelConfig, 'NUM_EVIDENCE_TASKS', 3)
        self.task_evidence_dim = getattr(ModelConfig, 'EVIDENCE_DIM_PER_TASK', 64) * \
            self.num_evidence_tasks
        self.use_missing_evidence_predictor = getattr(ModelConfig, 'USE_MISSING_EVIDENCE_PREDICTOR', True)
        self.missing_evidence_predictor_type = getattr(ModelConfig, 'MISSING_EVIDENCE_PREDICTOR_TYPE', 'mlp')
        self.missing_evidence_hidden_dim = getattr(ModelConfig, 'MISSING_EVIDENCE_HIDDEN_DIM', 256)
        self.use_shared_missing_evidence_predictor = getattr(ModelConfig, 'USE_SHARED_MISSING_EVIDENCE_PREDICTOR', True)
        self.use_evidence_residual_fusion = getattr(ModelConfig, 'USE_EVIDENCE_RESIDUAL_FUSION', True)
        self.evidence_residual_type = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_TYPE', 'bias')
        self.evidence_residual_hidden_dim = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_HIDDEN_DIM', 128)
        self.evidence_residual_scale = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_SCALE', 0.1)
        self.learnable_evidence_residual_scale = getattr(
            ModelConfig, 'LEARNABLE_EVIDENCE_RESIDUAL_SCALE', False
        )
        self.use_completed_evidence_for_residual = getattr(
            ModelConfig, 'USE_COMPLETED_EVIDENCE_FOR_RESIDUAL', True
        )
        # TE-DMoE: weighted missing-slot evidence completion.
        self.missing_evidence_mean_weight = getattr(ModelConfig, 'MISSING_EVIDENCE_MEAN_WEIGHT', 0.7)
        self.missing_evidence_pred_weight = getattr(ModelConfig, 'MISSING_EVIDENCE_PRED_WEIGHT', 0.3)
        self.evidence_residual_gamma_tanh = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_GAMMA_TANH', True)
        self.enable_evidence_residual_fusion = (
            self.use_task_evidence and
            self.use_missing_evidence_predictor and
            self.use_evidence_residual_fusion
        )
        self.n_modalities = 4
        self.missing_evidence_predictor = None
        self.evidence_residual_fusion = None
        if self.use_task_evidence and self.use_missing_evidence_predictor:
            # TE-DMoE: shared lightweight predictor for missing modality evidence.
            self.missing_evidence_predictor = MissingEvidencePredictor(
                evidence_dim=self.task_evidence_dim,
                num_modalities=self.n_modalities,
                hidden_dim=self.missing_evidence_hidden_dim,
                predictor_type=self.missing_evidence_predictor_type,
                use_shared_predictor=self.use_shared_missing_evidence_predictor,
                use_learnable_scale=getattr(ModelConfig, 'MISSING_EVIDENCE_USE_LEARNABLE_SCALE', True),
                init_scale=getattr(ModelConfig, 'MISSING_EVIDENCE_INIT_SCALE', 1.0),
                num_tasks=self.num_evidence_tasks,
                use_task_prototype_query=getattr(ModelConfig, 'USE_TASK_PROTOTYPE_QUERY', True),
            ).cuda()
        if self.enable_evidence_residual_fusion:
            # TE-DMoE: lightweight completed-evidence residual branch to safely perturb final logits.
            self.evidence_residual_fusion = EvidenceResidualFusion(
                evidence_dim=self.task_evidence_dim,
                num_classes=ModelConfig.N_CLASSES,
                hidden_dim=self.evidence_residual_hidden_dim,
                residual_type=self.evidence_residual_type,
                residual_scale=self.evidence_residual_scale,
                learnable_scale=self.learnable_evidence_residual_scale,
                gamma_tanh=self.evidence_residual_gamma_tanh
            ).cuda()

        self.expert_ls = nn.ModuleList([
            self._build_single_expert(0),
            self._build_single_expert(1),
            self._build_single_expert(2),
            self._build_single_expert(3)
        ])
        self.router = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=4 * 3),
        ).cuda()

        # self.router = nn.Sequential(
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=5, stride=2, padding=2),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=5, stride=2, padding=2),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=5, stride=2, padding=2),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=5, stride=2, padding=2),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=5, stride=2, padding=2),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Flatten(),
        #     nn.Linear(in_features=256, out_features=64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=64, out_features=4 * 3),
        # ).cuda()

        # self.router = nn.Sequential(
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=4, out_channels=4, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
        #     nn.ReLU(inplace=True),
        #     nn.Flatten(),
        #     nn.Linear(in_features=256, out_features=64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=64, out_features=4 * 3),
        # ).cuda()

        self.expert_output_shape = self.expert_ls[0](torch.rand(1, 1, 128, 128, 128).cuda()).shape

    def _build_single_expert(self, expert_id):
        net = UNet(
            input_channels=1,
            n_classes=ModelConfig.N_CLASSES,
            n_stages=ModelConfig.N_STAGES,
            n_features_per_stage=ModelConfig.N_FEATURES_PER_STAGE,
            kernel_size=ModelConfig.KERNEL_SIZES,
            strides=ModelConfig.STRIDES,
            # TE-DMoE: optional evidence head for modality experts.
            use_task_evidence=getattr(ModelConfig, 'USE_TASK_EVIDENCE', False),
            evidence_dim_per_task=getattr(ModelConfig, 'EVIDENCE_DIM_PER_TASK', 64),
            num_evidence_tasks=getattr(ModelConfig, 'NUM_EVIDENCE_TASKS', 3),
            evidence_pooling=getattr(ModelConfig, 'EVIDENCE_POOLING', 'avg'),
            use_task_conditioned_evidence=getattr(ModelConfig, 'USE_TASK_CONDITIONED_EVIDENCE', True),
            task_evidence_mask_source=getattr(ModelConfig, 'TASK_EVIDENCE_MASK_SOURCE', 'gt_then_pred'),
            task_evidence_use_gt_mask_in_train=getattr(
                ModelConfig, 'TASK_EVIDENCE_USE_GT_MASK_IN_TRAIN', True
            ),
            task_evidence_use_pred_mask_in_eval=getattr(
                ModelConfig, 'TASK_EVIDENCE_USE_PRED_MASK_IN_EVAL', True
            ),
            evidence_num_tokens=getattr(ModelConfig, 'EVIDENCE_NUM_TOKENS', 2),
            use_hierarchical_evidence=getattr(ModelConfig, 'USE_HIERARCHICAL_EVIDENCE', True),
            hierarchical_region_names=getattr(
                ModelConfig, 'HIERARCHICAL_REGION_NAMES', ("R1_ET", "R2_TC_MINUS_ET", "R3_WT_MINUS_TC")
            )
        ).cuda()

        if self.pretrained_expert_file_list[expert_id] is not None:
            # TE-DMoE: safe partial loading for old/new expert checkpoints.
            expert_state_dict = torch.load(self.pretrained_expert_file_list[expert_id])
            _load_partial_state_dict_with_report(
                net,
                expert_state_dict,
                module_name=f'DMoMEOutputLevel.expert_{expert_id}',
                skip_prefixes=_get_expert_partial_skip_prefixes()
            )

        return net

    def forward(self, x, return_aux=False, evidence_labels=None):
        modality_mask = (x == 0).all(dim=-1).all(dim=-1).all(dim=-1) # True if modality missing

        weight = self.router(x).view(-1, 4, 3, 1, 1, 1)
        weight[modality_mask] = float('-inf')
        weight = nn.functional.softmax(weight, dim=1)

        collect_task_evidence = self.use_task_evidence and (return_aux or self.enable_evidence_residual_fusion)
        if not collect_task_evidence:
            output = torch.stack([
                torch.cat([
                    torch.zeros(self.expert_output_shape).cuda()
                    if modality_mask[sample_idx, modality_idx]
                    else self.expert_ls[modality_idx](x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...])
                    for sample_idx in range(x.shape[0])
                ], dim=0)
                for modality_idx in range(4)
            ], dim=1)

            output = output * weight
            output = output.sum(dim=1)

            return output

        # TE-DMoE: collect per-modality evidence vectors aligned with missing-modality mask.
        modality_output_ls = []
        modality_evidence_ls = []
        zero_expert_output = x.new_zeros(self.expert_output_shape)
        zero_task_evidence = x.new_zeros((1, self.task_evidence_dim))
        for modality_idx in range(4):
            cur_modality_output_ls = []
            cur_modality_evidence_ls = []
            for sample_idx in range(x.shape[0]):
                if modality_mask[sample_idx, modality_idx]:
                    cur_modality_output_ls.append(zero_expert_output)
                    cur_modality_evidence_ls.append(zero_task_evidence)
                    continue

                cur_evidence_labels = evidence_labels[sample_idx:sample_idx + 1, ...] \
                    if evidence_labels is not None else None
                cur_expert_output, cur_task_evidence = self.expert_ls[modality_idx](
                    x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...],
                    return_evidence=True,
                    evidence_labels=cur_evidence_labels
                )
                cur_modality_output_ls.append(cur_expert_output)
                cur_modality_evidence_ls.append(
                    zero_task_evidence if cur_task_evidence is None else cur_task_evidence
                )

            modality_output_ls.append(torch.cat(cur_modality_output_ls, dim=0))
            modality_evidence_ls.append(torch.cat(cur_modality_evidence_ls, dim=0))

        output = torch.stack(modality_output_ls, dim=1)
        output = output * weight
        output = output.sum(dim=1)

        task_evidence = torch.stack(modality_evidence_ls, dim=1)
        predicted_task_evidence, mean_task_evidence = _predict_missing_task_evidence(
            self.missing_evidence_predictor, task_evidence, modality_mask, modality_reliability=None
        )
        evidence_aux = _build_task_evidence_aux(
            task_evidence,
            modality_mask,
            predicted_task_evidence=predicted_task_evidence,
            mean_task_evidence=mean_task_evidence,
            missing_fusion_mean_weight=self.missing_evidence_mean_weight,
            missing_fusion_pred_weight=self.missing_evidence_pred_weight
        )
        final_output, evidence_residual_logits, evidence_residual_gamma, evidence_residual_beta, evidence_residual_scale = \
            _apply_task_evidence_residual(
                self.evidence_residual_fusion,
                output,
                evidence_aux,
                use_completed_evidence_for_residual=self.use_completed_evidence_for_residual
            )
        if not return_aux:
            return final_output

        return dict(
            logits=final_output,
            original_logits=output,
            evidence_residual_logits=evidence_residual_logits,
            evidence_residual_gamma=evidence_residual_gamma,
            evidence_residual_beta=evidence_residual_beta,
            evidence_residual_bias=evidence_residual_beta,
            evidence_residual_scale=evidence_residual_scale,
            **evidence_aux
        )


class DMoMEProbLevel(nn.Module):
    def __init__(self, ignore_assert=False):
        super().__init__()
        if ignore_assert:
            print("CAREFUL!!!!!!YOU ARE IGNORING THE ASSERTATION OF MODEL CREATION!!!!!!")
        else:
            assert not ModelConfig.TRAIN_LOSS_ARGS['need_sigmoid'] and not ModelConfig.VAL_LOSS_ARGS['need_sigmoid'], \
                "loss for this model does not need sigmoid!"

        self.pretrained_expert_file_list = ModelConfig.PRETRAINED_EXPERT_FILE_LIST
        self.n_stages = ModelConfig.N_STAGES
        # TE-DMoE: control whether joint model collects expert evidence tensors.
        self.use_task_evidence = getattr(ModelConfig, 'USE_TASK_EVIDENCE', False)
        self.num_evidence_tasks = getattr(ModelConfig, 'NUM_EVIDENCE_TASKS', 3)
        self.task_evidence_dim = getattr(ModelConfig, 'EVIDENCE_DIM_PER_TASK', 64) * \
            self.num_evidence_tasks
        self.use_missing_evidence_predictor = getattr(ModelConfig, 'USE_MISSING_EVIDENCE_PREDICTOR', True)
        self.missing_evidence_predictor_type = getattr(ModelConfig, 'MISSING_EVIDENCE_PREDICTOR_TYPE', 'mlp')
        self.missing_evidence_hidden_dim = getattr(ModelConfig, 'MISSING_EVIDENCE_HIDDEN_DIM', 256)
        self.use_shared_missing_evidence_predictor = getattr(ModelConfig, 'USE_SHARED_MISSING_EVIDENCE_PREDICTOR', True)
        self.use_evidence_residual_fusion = getattr(ModelConfig, 'USE_EVIDENCE_RESIDUAL_FUSION', True)
        self.evidence_residual_type = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_TYPE', 'bias')
        self.evidence_residual_hidden_dim = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_HIDDEN_DIM', 128)
        self.evidence_residual_scale = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_SCALE', 0.1)
        self.learnable_evidence_residual_scale = getattr(
            ModelConfig, 'LEARNABLE_EVIDENCE_RESIDUAL_SCALE', False
        )
        self.use_completed_evidence_for_residual = getattr(
            ModelConfig, 'USE_COMPLETED_EVIDENCE_FOR_RESIDUAL', True
        )
        # TE-DMoE: weighted missing-slot evidence completion.
        self.missing_evidence_mean_weight = getattr(ModelConfig, 'MISSING_EVIDENCE_MEAN_WEIGHT', 0.7)
        self.missing_evidence_pred_weight = getattr(ModelConfig, 'MISSING_EVIDENCE_PRED_WEIGHT', 0.3)
        self.evidence_residual_gamma_tanh = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_GAMMA_TANH', True)
        self.enable_evidence_residual_fusion = (
            self.use_task_evidence and
            self.use_missing_evidence_predictor and
            self.use_evidence_residual_fusion
        )
        self.n_modalities = 4
        self.missing_evidence_predictor = None
        self.evidence_residual_fusion = None
        if self.use_task_evidence and self.use_missing_evidence_predictor:
            # TE-DMoE: shared lightweight predictor for missing modality evidence.
            self.missing_evidence_predictor = MissingEvidencePredictor(
                evidence_dim=self.task_evidence_dim,
                num_modalities=self.n_modalities,
                hidden_dim=self.missing_evidence_hidden_dim,
                predictor_type=self.missing_evidence_predictor_type,
                use_shared_predictor=self.use_shared_missing_evidence_predictor,
                use_learnable_scale=getattr(ModelConfig, 'MISSING_EVIDENCE_USE_LEARNABLE_SCALE', True),
                init_scale=getattr(ModelConfig, 'MISSING_EVIDENCE_INIT_SCALE', 1.0),
            ).cuda()
        if self.enable_evidence_residual_fusion:
            # TE-DMoE: lightweight completed-evidence residual branch to safely perturb final logits.
            self.evidence_residual_fusion = EvidenceResidualFusion(
                evidence_dim=self.task_evidence_dim,
                num_classes=ModelConfig.N_CLASSES,
                hidden_dim=self.evidence_residual_hidden_dim,
                residual_type=self.evidence_residual_type,
                residual_scale=self.evidence_residual_scale,
                learnable_scale=self.learnable_evidence_residual_scale,
                gamma_tanh=self.evidence_residual_gamma_tanh
            ).cuda()

        self.expert_ls = nn.ModuleList([
            self._build_single_expert(0),
            self._build_single_expert(1),
            self._build_single_expert(2),
            self._build_single_expert(3)
        ])
        self.router = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=4 * 3),
        ).cuda()

        self.expert_output_shape = self.expert_ls[0](torch.rand(1, 1, 128, 128, 128).cuda()).shape

    def _build_single_expert(self, expert_id):
        net = UNet(
            input_channels=1,
            n_classes=ModelConfig.N_CLASSES,
            n_stages=ModelConfig.N_STAGES,
            n_features_per_stage=ModelConfig.N_FEATURES_PER_STAGE,
            kernel_size=ModelConfig.KERNEL_SIZES,
            strides=ModelConfig.STRIDES,
            # TE-DMoE: optional evidence head for modality experts.
            use_task_evidence=getattr(ModelConfig, 'USE_TASK_EVIDENCE', False),
            evidence_dim_per_task=getattr(ModelConfig, 'EVIDENCE_DIM_PER_TASK', 64),
            num_evidence_tasks=getattr(ModelConfig, 'NUM_EVIDENCE_TASKS', 3),
            evidence_pooling=getattr(ModelConfig, 'EVIDENCE_POOLING', 'avg'),
            use_task_conditioned_evidence=getattr(ModelConfig, 'USE_TASK_CONDITIONED_EVIDENCE', True),
            task_evidence_mask_source=getattr(ModelConfig, 'TASK_EVIDENCE_MASK_SOURCE', 'gt_then_pred'),
            task_evidence_use_gt_mask_in_train=getattr(
                ModelConfig, 'TASK_EVIDENCE_USE_GT_MASK_IN_TRAIN', True
            ),
            task_evidence_use_pred_mask_in_eval=getattr(
                ModelConfig, 'TASK_EVIDENCE_USE_PRED_MASK_IN_EVAL', True
            ),
            evidence_num_tokens=getattr(ModelConfig, 'EVIDENCE_NUM_TOKENS', 2),
            use_hierarchical_evidence=getattr(ModelConfig, 'USE_HIERARCHICAL_EVIDENCE', True),
            hierarchical_region_names=getattr(
                ModelConfig, 'HIERARCHICAL_REGION_NAMES', ("R1_ET", "R2_TC_MINUS_ET", "R3_WT_MINUS_TC")
            )
        ).cuda()

        if self.pretrained_expert_file_list[expert_id] is not None:
            # TE-DMoE: safe partial loading for old/new expert checkpoints.
            expert_state_dict = torch.load(self.pretrained_expert_file_list[expert_id])
            _load_partial_state_dict_with_report(
                net,
                expert_state_dict,
                module_name=f'DMoMEProbLevel.expert_{expert_id}',
                skip_prefixes=_get_expert_partial_skip_prefixes()
            )

        return net

    def forward(self, x, return_aux=False, evidence_labels=None):
        modality_mask = (x == 0).all(dim=-1).all(dim=-1).all(dim=-1)

        weight = self.router(x).view(-1, 4, 3, 1, 1, 1)
        weight[modality_mask] = float('-inf')
        weight = nn.functional.softmax(weight, dim=1)

        collect_task_evidence = self.use_task_evidence and (return_aux or self.enable_evidence_residual_fusion)
        if not collect_task_evidence:
            output = torch.stack([
                torch.cat([
                    torch.zeros(self.expert_output_shape).cuda()
                    if modality_mask[sample_idx, modality_idx]
                    else torch.sigmoid(
                        self.expert_ls[modality_idx](x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...])
                    )
                    for sample_idx in range(x.shape[0])
                ], dim=0)
                for modality_idx in range(4)
            ], dim=1)

            output = output * weight
            output = torch.clamp(output.sum(dim=1), min=0.0, max=1.0)

            return output

        # TE-DMoE: collect per-modality evidence vectors aligned with missing-modality mask.
        modality_output_ls = []
        modality_evidence_ls = []
        zero_expert_output = x.new_zeros(self.expert_output_shape)
        zero_task_evidence = x.new_zeros((1, self.task_evidence_dim))
        for modality_idx in range(4):
            cur_modality_output_ls = []
            cur_modality_evidence_ls = []
            for sample_idx in range(x.shape[0]):
                if modality_mask[sample_idx, modality_idx]:
                    cur_modality_output_ls.append(zero_expert_output)
                    cur_modality_evidence_ls.append(zero_task_evidence)
                    continue

                cur_evidence_labels = evidence_labels[sample_idx:sample_idx + 1, ...] \
                    if evidence_labels is not None else None
                cur_expert_logits, cur_task_evidence = self.expert_ls[modality_idx](
                    x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...],
                    return_evidence=True,
                    evidence_labels=cur_evidence_labels
                )
                cur_modality_output_ls.append(torch.sigmoid(cur_expert_logits))
                cur_modality_evidence_ls.append(
                    zero_task_evidence if cur_task_evidence is None else cur_task_evidence
                )

            modality_output_ls.append(torch.cat(cur_modality_output_ls, dim=0))
            modality_evidence_ls.append(torch.cat(cur_modality_evidence_ls, dim=0))

        output = torch.stack(modality_output_ls, dim=1)
        output = output * weight
        output = torch.clamp(output.sum(dim=1), min=0.0, max=1.0)

        task_evidence = torch.stack(modality_evidence_ls, dim=1)
        predicted_task_evidence, mean_task_evidence = _predict_missing_task_evidence(
            self.missing_evidence_predictor, task_evidence, modality_mask, modality_reliability=None
        )
        evidence_aux = _build_task_evidence_aux(
            task_evidence,
            modality_mask,
            predicted_task_evidence=predicted_task_evidence,
            mean_task_evidence=mean_task_evidence,
            missing_fusion_mean_weight=self.missing_evidence_mean_weight,
            missing_fusion_pred_weight=self.missing_evidence_pred_weight
        )
        final_output, evidence_residual_logits, evidence_residual_gamma, evidence_residual_beta, evidence_residual_scale = \
            _apply_task_evidence_residual(
                self.evidence_residual_fusion,
                output,
                evidence_aux,
                use_completed_evidence_for_residual=self.use_completed_evidence_for_residual
            )
        if not return_aux:
            return final_output

        return dict(
            logits=final_output,
            original_logits=output,
            evidence_residual_logits=evidence_residual_logits,
            evidence_residual_gamma=evidence_residual_gamma,
            evidence_residual_beta=evidence_residual_beta,
            evidence_residual_bias=evidence_residual_beta,
            evidence_residual_scale=evidence_residual_scale,
            **evidence_aux
        )


class ExpertNet(UNet):
    def __init__(self, input_channels, n_classes, n_stages, n_features_per_stage, kernel_size, strides,
                 use_task_evidence=False, evidence_dim_per_task=64, num_evidence_tasks=3, evidence_pooling='avg',
                 use_task_conditioned_evidence=True, task_evidence_mask_source='gt_then_pred',
                 task_evidence_use_gt_mask_in_train=True, task_evidence_use_pred_mask_in_eval=True,
                 evidence_num_tokens=2,
                 use_hierarchical_evidence=True,
                 hierarchical_region_names=("R1_ET", "R2_TC_MINUS_ET", "R3_WT_MINUS_TC")):
        super().__init__(
            input_channels=input_channels,
            n_classes=n_classes,
            n_stages=n_stages,
            n_features_per_stage=n_features_per_stage,
            kernel_size=kernel_size,
            strides=strides,
            use_task_evidence=use_task_evidence,
            evidence_dim_per_task=evidence_dim_per_task,
            num_evidence_tasks=num_evidence_tasks,
            evidence_pooling=evidence_pooling,
            use_task_conditioned_evidence=use_task_conditioned_evidence,
            task_evidence_mask_source=task_evidence_mask_source,
            task_evidence_use_gt_mask_in_train=task_evidence_use_gt_mask_in_train,
            task_evidence_use_pred_mask_in_eval=task_evidence_use_pred_mask_in_eval,
            evidence_num_tokens=evidence_num_tokens,
            use_hierarchical_evidence=use_hierarchical_evidence,
            hierarchical_region_names=hierarchical_region_names
        )

    def forward(self, x, return_evidence=False, evidence_labels=None, return_evidence_aux=False):
        encoded_feat_maps = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoded_feat_maps.append(x)

        low_res_input = encoded_feat_maps[-1]
        for i in range(len(self.decoder_stages)):
            output = self.connect_layers[i](low_res_input)
            output = torch.cat((output, encoded_feat_maps[-i - 2]), dim=1)
            output = self.decoder_stages[i](output)
            low_res_input = output

        # TE-DMoE: expose optional expert evidence without changing default behavior.
        if return_evidence:
            evidence_aux = None
            if self.use_task_evidence:
                # TE-DMoE: feature-level expert uses its segmentation head only for task-mask extraction.
                seg_logits_for_mask = self.seg_layers[-1](output)
                if return_evidence_aux:
                    evidence, evidence_aux = self._extract_task_evidence(
                        encoded_feat_maps[-1],
                        seg_logits=seg_logits_for_mask,
                        evidence_labels=evidence_labels,
                        return_aux=True
                    )
                else:
                    evidence = self._extract_task_evidence(
                        encoded_feat_maps[-1],
                        seg_logits=seg_logits_for_mask,
                        evidence_labels=evidence_labels
                    )
            else:
                evidence = None
            if return_evidence_aux:
                return output, evidence, evidence_aux
            return output, evidence

        return output


class DMoMEFeatureLevel(nn.Module):
    def __init__(self, ignore_assert=False):
        super().__init__()
        if ignore_assert:
            print("CAREFUL!!!!!!YOU ARE IGNORING THE ASSERTATION OF MODEL CREATION!!!!!!")
        else:
            assert ModelConfig.TRAIN_LOSS_ARGS['need_sigmoid'] and ModelConfig.VAL_LOSS_ARGS['need_sigmoid'], \
                "loss for this model needs sigmoid!"
        self.pretrained_expert_file_list = ModelConfig.PRETRAINED_EXPERT_FILE_LIST
        self.n_stages = ModelConfig.N_STAGES
        # TE-DMoE: control whether joint model collects expert evidence tensors.
        self.use_task_evidence = getattr(ModelConfig, 'USE_TASK_EVIDENCE', False)
        self.num_evidence_tasks = getattr(ModelConfig, 'NUM_EVIDENCE_TASKS', 3)
        self.evidence_dim_per_task = getattr(ModelConfig, 'EVIDENCE_DIM_PER_TASK', 64)
        self.task_evidence_dim = self.evidence_dim_per_task * self.num_evidence_tasks
        self.use_missing_evidence_predictor = getattr(ModelConfig, 'USE_MISSING_EVIDENCE_PREDICTOR', True)
        self.missing_evidence_predictor_type = getattr(ModelConfig, 'MISSING_EVIDENCE_PREDICTOR_TYPE', 'mlp')
        self.missing_evidence_hidden_dim = getattr(ModelConfig, 'MISSING_EVIDENCE_HIDDEN_DIM', 256)
        self.use_shared_missing_evidence_predictor = getattr(ModelConfig, 'USE_SHARED_MISSING_EVIDENCE_PREDICTOR', True)
        self.use_evidence_residual_fusion = getattr(ModelConfig, 'USE_EVIDENCE_RESIDUAL_FUSION', True)
        self.use_evidence_guided_decoder = getattr(ModelConfig, 'USE_EVIDENCE_GUIDED_DECODER', True)
        self.use_legacy_evidence_output_residual = getattr(
            ModelConfig, 'USE_LEGACY_EVIDENCE_OUTPUT_RESIDUAL', False
        )
        self.use_hierarchical_evidence = getattr(ModelConfig, 'USE_HIERARCHICAL_EVIDENCE', True)
        self.use_hierarchical_output_composition = getattr(
            ModelConfig, 'USE_HIERARCHICAL_OUTPUT_COMPOSITION', True
        )
        self.hierarchical_region_names = getattr(
            ModelConfig, 'HIERARCHICAL_REGION_NAMES', ("R1_ET", "R2_TC_MINUS_ET", "R3_WT_MINUS_TC")
        )
        self.evidence_guided_decoder_hidden_dim = getattr(ModelConfig, 'EVIDENCE_GUIDED_DECODER_HIDDEN_DIM', 128)
        self.evidence_residual_type = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_TYPE', 'bias')
        self.evidence_residual_hidden_dim = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_HIDDEN_DIM', 128)
        self.evidence_residual_scale = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_SCALE', 0.1)
        self.learnable_evidence_residual_scale = getattr(
            ModelConfig, 'LEARNABLE_EVIDENCE_RESIDUAL_SCALE', False
        )
        self.use_completed_evidence_for_residual = getattr(
            ModelConfig, 'USE_COMPLETED_EVIDENCE_FOR_RESIDUAL', True
        )
        # TE-DMoE: weighted missing-slot evidence completion.
        self.missing_evidence_mean_weight = getattr(ModelConfig, 'MISSING_EVIDENCE_MEAN_WEIGHT', 0.7)
        self.missing_evidence_pred_weight = getattr(ModelConfig, 'MISSING_EVIDENCE_PRED_WEIGHT', 0.3)
        # TE-DMoE: plain + hierarchical residual decoder scaling.
        self.evidence_hier_residual_scale = getattr(ModelConfig, 'EVIDENCE_HIER_RESIDUAL_SCALE', 1.0)
        self.evidence_residual_gamma_tanh = getattr(ModelConfig, 'EVIDENCE_RESIDUAL_GAMMA_TANH', True)
        self.use_task_prototype_anchor = getattr(ModelConfig, 'USE_TASK_PROTOTYPE_ANCHOR', True)
        self.prototype_ema_momentum = getattr(ModelConfig, 'PROTOTYPE_EMA_MOMENTUM', 0.95)
        # TE-DMoE: semantic router over expert feature summaries (+ availability + optional uncertainty).
        self.use_semantic_router = getattr(ModelConfig, 'USE_SEMANTIC_ROUTER', True)
        self.semantic_router_hidden_dim = getattr(ModelConfig, 'SEMANTIC_ROUTER_HIDDEN_DIM', 128)
        self.semantic_router_use_uncertainty = getattr(ModelConfig, 'SEMANTIC_ROUTER_USE_UNCERTAINTY', True)
        self.semantic_router_uncertainty_lambda = getattr(ModelConfig, 'SEMANTIC_ROUTER_UNCERTAINTY_LAMBDA', 1.0)
        self.semantic_router_eps = getattr(ModelConfig, 'SEMANTIC_ROUTER_EPS', 1e-8)
        # TE-DMoE: task-specific semantic routing (per task modality weights).
        self.use_task_specific_router = getattr(ModelConfig, 'USE_TASK_SPECIFIC_ROUTER', True)
        self.task_router_proto_dim = getattr(
            ModelConfig, 'TASK_ROUTER_PROTO_DIM', getattr(ModelConfig, 'EVIDENCE_DIM_PER_TASK', 64)
        )
        self.task_router_use_decomposed_logit = getattr(ModelConfig, 'TASK_ROUTER_USE_DECOMPOSED_LOGIT', True)
        self.task_router_beta_init = getattr(ModelConfig, 'TASK_ROUTER_BETA_INIT', 1.0)
        # TE-DMoE: bottleneck evidence FiLM controls.
        self.use_evidence_film = getattr(ModelConfig, 'USE_EVIDENCE_FILM', True)
        self.use_evidence_spatial_attention = getattr(ModelConfig, 'USE_EVIDENCE_SPATIAL_ATTENTION', True)
        self.film_hidden_dim = getattr(ModelConfig, 'FILM_HIDDEN_DIM', 128)
        self.film_zero_init = getattr(ModelConfig, 'FILM_ZERO_INIT', True)
        self.film_alpha_init = getattr(ModelConfig, 'FILM_ALPHA_INIT', 0.0)
        self.film_spatial_attention_channels = getattr(ModelConfig, 'FILM_SPATIAL_ATTENTION_CHANNELS', 64)
        # TE-DMoE: main evidence-guided spatial modulation path.
        self.use_task_spatial_evidence_fusion = getattr(ModelConfig, 'USE_TASK_SPATIAL_EVIDENCE_FUSION', True)
        self.task_spatial_fusion_hidden_dim = getattr(ModelConfig, 'TASK_SPATIAL_FUSION_HIDDEN_DIM', 64)
        self.task_spatial_fusion_zero_init = getattr(ModelConfig, 'TASK_SPATIAL_FUSION_ZERO_INIT', True)
        self.task_spatial_fusion_alpha_init = getattr(
            ModelConfig, 'GATE_ALPHA_INIT',
            getattr(ModelConfig, 'TASK_SPATIAL_FUSION_ALPHA_INIT', 0.1)
        )
        self.task_spatial_fusion_gate_bias_init = getattr(
            ModelConfig, 'GATE_BIAS_INIT',
            getattr(ModelConfig, 'TASK_SPATIAL_FUSION_GATE_BIAS_INIT', -2.0)
        )
        # TE-DMoE: optional evidence-attention alignment supervision support.
        self.use_task_evidence_alignment_loss = getattr(ModelConfig, 'USE_TASK_EVIDENCE_ALIGNMENT_LOSS', False)
        # TE-DMoE: two-stage training runtime switch (teacher first, then completion+fusion).
        self.runtime_stage1_teacher_mode = False
        self.enable_evidence_residual_fusion = (
            self.use_task_evidence and
            self.use_missing_evidence_predictor and
            self.use_evidence_residual_fusion
        )
        self.enable_evidence_guided_decoder = (
            self.use_task_evidence and self.use_evidence_guided_decoder
        )
        self.enable_legacy_evidence_output_residual = (
            self.enable_evidence_guided_decoder and self.use_legacy_evidence_output_residual
        )
        self.n_modalities = 4
        self.missing_evidence_predictor = None
        self.evidence_residual_fusion = None
        self.task_prototype_bank = None
        self.task_conditioned_decoder = None
        self.evidence_film = None
        self.task_spatial_evidence_fusion = None
        if self.use_task_evidence and self.use_missing_evidence_predictor:
            # TE-DMoE: shared lightweight predictor for missing modality evidence.
            self.missing_evidence_predictor = MissingEvidencePredictor(
                evidence_dim=self.task_evidence_dim,
                num_modalities=self.n_modalities,
                hidden_dim=self.missing_evidence_hidden_dim,
                predictor_type=self.missing_evidence_predictor_type,
                use_shared_predictor=self.use_shared_missing_evidence_predictor,
                use_learnable_scale=getattr(ModelConfig, 'MISSING_EVIDENCE_USE_LEARNABLE_SCALE', True),
                init_scale=getattr(ModelConfig, 'MISSING_EVIDENCE_INIT_SCALE', 1.0),
                num_tasks=self.num_evidence_tasks,
                use_task_prototype_query=getattr(ModelConfig, 'USE_TASK_PROTOTYPE_QUERY', True),
            ).cuda()
        if self.enable_evidence_residual_fusion:
            # TE-DMoE: lightweight completed-evidence residual branch to safely perturb final logits.
            self.evidence_residual_fusion = EvidenceResidualFusion(
                evidence_dim=self.task_evidence_dim,
                num_classes=ModelConfig.N_CLASSES,
                hidden_dim=self.evidence_residual_hidden_dim,
                residual_type=self.evidence_residual_type,
                residual_scale=self.evidence_residual_scale,
                learnable_scale=self.learnable_evidence_residual_scale,
                gamma_tanh=self.evidence_residual_gamma_tanh
            ).cuda()
        if self.use_task_evidence and self.use_task_prototype_anchor:
            # TE-DMoE: lightweight EMA task prototype anchors for hierarchical regions (R1/R2/R3).
            self.task_prototype_bank = TaskPrototypeBank(
                num_tasks=getattr(ModelConfig, 'NUM_EVIDENCE_TASKS', 3),
                evidence_dim_per_task=getattr(ModelConfig, 'EVIDENCE_DIM_PER_TASK', 64),
                momentum=self.prototype_ema_momentum
            ).cuda()

        self.expert_ls = nn.ModuleList([
            self._build_single_expert(0),
            self._build_single_expert(1),
            self._build_single_expert(2),
            self._build_single_expert(3)
        ])
        self.router = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=4),
        ).cuda()
        self.segmentation_head = nn.Conv3d(
            in_channels=self.expert_ls[0].seg_layers[-1].in_channels,
            out_channels=self.expert_ls[0].seg_layers[-1].out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.router_feature_dim = self.expert_ls[0].seg_layers[-1].in_channels
        self.base_router_input_dim = self.router_feature_dim + 1 + (
            1 if self.semantic_router_use_uncertainty else 0
        )
        self.semantic_router_global_input_dim = self.base_router_input_dim + self.task_router_proto_dim
        if self.use_task_specific_router:
            self.task_router_prototypes = nn.Parameter(
                torch.randn(self.num_evidence_tasks, self.task_router_proto_dim) * 0.02
            )
            self.task_router_modality_embedding = nn.Embedding(self.n_modalities, self.task_router_proto_dim)
            self.task_router_beta = nn.Parameter(
                torch.full((1,), float(self.task_router_beta_init), dtype=torch.float32)
            )
        else:
            self.register_parameter('task_router_prototypes', None)
            self.task_router_modality_embedding = None
            self.register_parameter('task_router_beta', None)
        self.semantic_router_global_mlp = nn.Sequential(
            nn.LayerNorm(self.semantic_router_global_input_dim),
            nn.Linear(self.semantic_router_global_input_dim, self.semantic_router_hidden_dim),
            nn.GELU(),
            nn.Linear(self.semantic_router_hidden_dim, 1),
        )
        task_router_input_dim = (
            self.evidence_dim_per_task +
            self.task_router_proto_dim +
            self.task_router_proto_dim +
            (1 if self.semantic_router_use_uncertainty else 0)
        )
        self.semantic_router_task_mlp = nn.Sequential(
            nn.LayerNorm(task_router_input_dim),
            nn.Linear(task_router_input_dim, self.semantic_router_hidden_dim),
            nn.GELU(),
            nn.Linear(self.semantic_router_hidden_dim, 1),
        )
        self.semantic_router_task_input_dim = task_router_input_dim
        self.semantic_router_uncertainty_head = nn.Linear(self.router_feature_dim, 1) \
            if self.semantic_router_use_uncertainty else None
        if self.use_task_evidence and self.use_evidence_film:
            # TE-DMoE: strong evidence coupling at bottleneck with FiLM + spatial attention.
            self.evidence_film = EvidenceFiLMBottleneck(
                feature_dim=self.router_feature_dim,
                evidence_dim=self.task_evidence_dim,
                hidden_dim=self.film_hidden_dim,
                use_spatial_attention=self.use_evidence_spatial_attention,
                spatial_attention_channels=self.film_spatial_attention_channels,
                zero_init=self.film_zero_init,
                alpha_init=self.film_alpha_init
            ).cuda()
        if self.use_task_evidence and self.use_task_spatial_evidence_fusion:
            self.task_spatial_evidence_fusion = TaskSpatialEvidenceFusion(
                feature_dim=self.router_feature_dim,
                evidence_dim_per_task=getattr(ModelConfig, 'EVIDENCE_DIM_PER_TASK', 64),
                num_tasks=self.num_evidence_tasks,
                hidden_dim=self.task_spatial_fusion_hidden_dim,
                zero_init=self.task_spatial_fusion_zero_init,
                alpha_init=self.task_spatial_fusion_alpha_init,
                gate_bias_init=self.task_spatial_fusion_gate_bias_init,
                hierarchical_region_names=self.hierarchical_region_names,
            ).cuda()
        if self.enable_legacy_evidence_output_residual:
            # TE-DMoE: evidence-centric decoding with task-wise FiLM modulation.
            self.task_conditioned_decoder = TaskConditionedDecoder(
                feature_dim=self.expert_ls[0].seg_layers[-1].in_channels,
                evidence_dim_per_task=getattr(ModelConfig, 'EVIDENCE_DIM_PER_TASK', 64),
                num_tasks=getattr(ModelConfig, 'NUM_EVIDENCE_TASKS', 3),
                hidden_dim=self.evidence_guided_decoder_hidden_dim,
                use_hierarchical_output_composition=self.use_hierarchical_output_composition,
                hierarchical_region_names=self.hierarchical_region_names
            ).cuda()

        self.expert_output_shape = self.expert_ls[0](torch.rand(1, 1, 128, 128, 128).cuda()).shape

    def _build_single_expert(self, expert_id):
        net = ExpertNet(
            input_channels=1,
            n_classes=ModelConfig.N_CLASSES,
            n_stages=ModelConfig.N_STAGES,
            n_features_per_stage=ModelConfig.N_FEATURES_PER_STAGE,
            kernel_size=ModelConfig.KERNEL_SIZES,
            strides=ModelConfig.STRIDES,
            # TE-DMoE: optional evidence head for modality experts.
            use_task_evidence=getattr(ModelConfig, 'USE_TASK_EVIDENCE', False),
            evidence_dim_per_task=getattr(ModelConfig, 'EVIDENCE_DIM_PER_TASK', 64),
            num_evidence_tasks=getattr(ModelConfig, 'NUM_EVIDENCE_TASKS', 3),
            evidence_pooling=getattr(ModelConfig, 'EVIDENCE_POOLING', 'avg'),
            use_task_conditioned_evidence=getattr(ModelConfig, 'USE_TASK_CONDITIONED_EVIDENCE', True),
            task_evidence_mask_source=getattr(ModelConfig, 'TASK_EVIDENCE_MASK_SOURCE', 'gt_then_pred'),
            task_evidence_use_gt_mask_in_train=getattr(
                ModelConfig, 'TASK_EVIDENCE_USE_GT_MASK_IN_TRAIN', True
            ),
            task_evidence_use_pred_mask_in_eval=getattr(
                ModelConfig, 'TASK_EVIDENCE_USE_PRED_MASK_IN_EVAL', True
            ),
            evidence_num_tokens=getattr(ModelConfig, 'EVIDENCE_NUM_TOKENS', 2),
            use_hierarchical_evidence=getattr(ModelConfig, 'USE_HIERARCHICAL_EVIDENCE', True),
            hierarchical_region_names=getattr(
                ModelConfig, 'HIERARCHICAL_REGION_NAMES', ("R1_ET", "R2_TC_MINUS_ET", "R3_WT_MINUS_TC")
            )
        ).cuda()

        if self.pretrained_expert_file_list[expert_id] is not None:
            # TE-DMoE: safe partial loading for old/new expert checkpoints.
            expert_state_dict = torch.load(self.pretrained_expert_file_list[expert_id])
            _load_partial_state_dict_with_report(
                net,
                expert_state_dict,
                module_name=f'DMoMEFeatureLevel.expert_{expert_id}',
                skip_prefixes=_get_expert_partial_skip_prefixes()
            )

        return net

    # TE-DMoE: update EMA task prototypes from full-reference evidence [B, M, E].
    @torch.no_grad()
    def update_task_prototypes(self, full_reference_evidence, modality_available_mask=None):
        if self.task_prototype_bank is None or full_reference_evidence is None:
            return
        num_tasks = getattr(ModelConfig, 'NUM_EVIDENCE_TASKS', 3)
        dim_per_task = full_reference_evidence.shape[-1] // num_tasks
        split_evidence = full_reference_evidence.view(
            full_reference_evidence.shape[0], full_reference_evidence.shape[1], num_tasks, dim_per_task
        )
        self.task_prototype_bank.update(split_evidence, modality_available_mask=modality_available_mask)

    # TE-DMoE: expose current task prototypes for anchor-alignment loss.
    def get_task_prototypes(self):
        if self.task_prototype_bank is None:
            return None
        return self.task_prototype_bank.get()

    # TE-DMoE: runtime stage switch for two-stage training.
    def set_runtime_teacher_mode(self, teacher_mode=False):
        self.runtime_stage1_teacher_mode = bool(teacher_mode)

    # TE-DMoE: numerically stable masked softmax with safe all-masked fallback.
    def _masked_softmax(self, logits, valid_mask, dim=1):
        mask = valid_mask.float()
        masked_logits = logits.masked_fill(mask <= 0.0, -1e9)
        max_logits = masked_logits.max(dim=dim, keepdim=True).values
        max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))
        exp_score = torch.exp(masked_logits - max_logits) * mask
        denom = exp_score.sum(dim=dim, keepdim=True)
        weights = exp_score / (denom + float(self.semantic_router_eps))
        all_masked = (mask.sum(dim=dim, keepdim=True) <= 0.0)
        if all_masked.any():
            uniform = torch.full_like(weights, 1.0 / float(weights.shape[dim]))
            weights = torch.where(all_masked.expand_as(weights), uniform, weights)
            denom = torch.where(all_masked, torch.ones_like(denom), denom)
        return weights, denom

    # TE-DMoE: compute feature-level routing weights with semantic summaries and availability mask.
    def _compute_feature_router_weights(self, x, modality_feature, modality_mask, task_evidence=None):
        # modality_feature: [B, M, C, H, W, D], modality_mask: [B, M] where True means missing.
        modality_available = (~modality_mask).float()

        if not self.use_semantic_router:
            router_logits = self.router(x)
            router_weight, router_denom = self._masked_softmax(router_logits, modality_available, dim=1)
            task_router_weight = router_weight.unsqueeze(-1).expand(-1, -1, self.num_evidence_tasks)
            task_router_logits = router_logits.unsqueeze(-1).expand(-1, -1, self.num_evidence_tasks)
            task_router_global_component = task_router_logits
            task_router_task_component = torch.zeros_like(task_router_logits)
            task_router_beta = router_logits.new_tensor(0.0)
            return (
                router_weight,
                task_router_weight,
                router_logits,
                None,
                router_logits,
                router_denom,
                task_router_logits,
                task_router_global_component,
                task_router_task_component,
                task_router_beta
            )

        feature_summary = modality_feature.mean(dim=(3, 4, 5))  # [B, M, C]
        availability = modality_available.unsqueeze(-1)  # [B, M, 1]
        router_input_ls = [feature_summary, availability]
        uncertainty = None
        if self.semantic_router_uncertainty_head is not None:
            raw_uncertainty = self.semantic_router_uncertainty_head(
                feature_summary.reshape(-1, self.router_feature_dim)
            ).view(feature_summary.shape[0], feature_summary.shape[1], 1)
            uncertainty = nn.functional.softplus(raw_uncertainty)
            router_input_ls.append(uncertainty)

        base_router_input = torch.cat(router_input_ls, dim=-1)  # [B, M, C+1(+1)]
        modality_ids = torch.arange(self.n_modalities, device=modality_feature.device)
        modality_embed = self.task_router_modality_embedding(modality_ids).view(1, self.n_modalities, -1).expand(
            modality_feature.shape[0], -1, -1
        ) if self.task_router_modality_embedding is not None else \
            base_router_input.new_zeros((base_router_input.shape[0], base_router_input.shape[1], self.task_router_proto_dim))
        global_router_input = torch.cat([base_router_input, modality_embed], dim=-1)
        global_component = self.semantic_router_global_mlp(
            global_router_input.reshape(-1, global_router_input.shape[-1])
        ).view(global_router_input.shape[0], global_router_input.shape[1])  # [B, M]
        if uncertainty is not None:
            global_component = global_component - float(self.semantic_router_uncertainty_lambda) * uncertainty.squeeze(-1)
        router_weight, router_denom = self._masked_softmax(global_component, modality_available, dim=1)
        router_score = global_component

        if not self.use_task_specific_router:
            task_router_weight = router_weight.unsqueeze(-1).expand(-1, -1, self.num_evidence_tasks)
            task_router_logits = global_component.unsqueeze(-1).expand(-1, -1, self.num_evidence_tasks)
            task_router_global_component = task_router_logits
            task_router_task_component = torch.zeros_like(task_router_logits)
            task_router_beta = global_component.new_tensor(0.0)
            return (
                router_weight,
                task_router_weight,
                global_component,
                uncertainty,
                router_score,
                router_denom,
                task_router_logits,
                task_router_global_component,
                task_router_task_component,
                task_router_beta
            )

        if task_evidence is None:
            task_evidence = modality_feature.new_zeros(
                (modality_feature.shape[0], modality_feature.shape[1], self.task_evidence_dim)
            )
        task_evidence_split = task_evidence.view(
            task_evidence.shape[0], task_evidence.shape[1], self.num_evidence_tasks, -1
        )  # [B, M, T, D]
        modality_embed_t = modality_embed.unsqueeze(2).expand(-1, -1, self.num_evidence_tasks, -1)
        task_proto = self.task_router_prototypes.view(1, 1, self.num_evidence_tasks, -1).expand(
            modality_feature.shape[0], self.n_modalities, -1, -1
        )
        task_input_ls = [task_evidence_split, task_proto, modality_embed_t]
        if uncertainty is not None:
            task_input_ls.append(uncertainty.unsqueeze(2).expand(-1, -1, self.num_evidence_tasks, -1))
        task_router_input = torch.cat(task_input_ls, dim=-1)
        task_component = self.semantic_router_task_mlp(
            task_router_input.reshape(-1, task_router_input.shape[-1])
        ).view(task_router_input.shape[0], task_router_input.shape[1], task_router_input.shape[2])  # [B, M, T]

        beta = self.task_router_beta.view(1, 1, 1) if self.task_router_beta is not None else \
            task_component.new_ones((1, 1, 1))
        if self.task_router_use_decomposed_logit:
            task_router_logits = global_component.unsqueeze(-1) + beta * task_component
            task_router_global_component = global_component.unsqueeze(-1).expand_as(task_component)
            task_router_task_component = task_component
        else:
            task_router_logits = task_component
            task_router_global_component = torch.zeros_like(task_component)
            task_router_task_component = task_component
            beta = task_component.new_tensor(1.0).view(1, 1, 1)
        if uncertainty is not None:
            # TE-DMoE: uncertainty is [B, M, 1]; convert to [B, M, 1] for task-logit broadcasting.
            uncertainty_penalty = uncertainty.squeeze(-1).unsqueeze(-1)
            task_router_logits = task_router_logits - float(self.semantic_router_uncertainty_lambda) * \
                uncertainty_penalty
        task_router_weight, _ = self._masked_softmax(
            task_router_logits, modality_available.unsqueeze(-1), dim=1
        )
        return (
            router_weight,
            task_router_weight,
            global_component,
            uncertainty,
            router_score,
            router_denom,
            task_router_logits,
            task_router_global_component,
            task_router_task_component,
            beta.squeeze()
        )

    def forward(self, x, return_aux=False, evidence_labels=None):
        # True if modality missing
        modality_mask = (x == 0).all(dim=-1).all(dim=-1).all(dim=-1)

        collect_task_evidence = self.use_task_evidence and (
            return_aux or self.enable_evidence_residual_fusion or
            (self.evidence_film is not None) or
            (self.task_spatial_evidence_fusion is not None) or
            self.enable_legacy_evidence_output_residual
        )
        collect_task_evidence_maps = collect_task_evidence and return_aux and self.use_task_evidence_alignment_loss

        # TE-DMoE: collect per-modality evidence vectors aligned with missing-modality mask.
        modality_output_ls = []
        modality_evidence_ls = []
        modality_attn_map_ls = []
        modality_target_mask_ls = []
        modality_evidence_map_ls = []
        zero_expert_output = x.new_zeros(self.expert_output_shape)
        zero_task_evidence = x.new_zeros((1, self.task_evidence_dim))
        template_task_attn_maps = None
        template_task_masks = None
        for modality_idx in range(4):
            cur_modality_output_ls = []
            cur_modality_evidence_ls = []
            cur_modality_attn_ls = []
            cur_modality_mask_ls = []
            cur_modality_evidence_map_ls = []
            for sample_idx in range(x.shape[0]):
                if modality_mask[sample_idx, modality_idx]:
                    cur_modality_output_ls.append(zero_expert_output)
                    if collect_task_evidence:
                        cur_modality_evidence_ls.append(zero_task_evidence)
                    if collect_task_evidence_maps:
                        cur_modality_attn_ls.append(None)
                        cur_modality_mask_ls.append(None)
                        cur_modality_evidence_map_ls.append(None)
                    continue

                if collect_task_evidence:
                    cur_evidence_labels = evidence_labels[sample_idx:sample_idx + 1, ...] \
                        if evidence_labels is not None else None
                    if collect_task_evidence_maps:
                        expert_return = self.expert_ls[modality_idx](
                            x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...],
                            return_evidence=True,
                            evidence_labels=cur_evidence_labels,
                            return_evidence_aux=True
                        )
                        if isinstance(expert_return, tuple) and len(expert_return) == 3:
                            cur_expert_output, cur_task_evidence, cur_task_evidence_aux = expert_return
                        else:
                            cur_expert_output, cur_task_evidence = expert_return
                            cur_task_evidence_aux = None
                    else:
                        cur_expert_output, cur_task_evidence = self.expert_ls[modality_idx](
                            x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...],
                            return_evidence=True,
                            evidence_labels=cur_evidence_labels
                        )
                        cur_task_evidence_aux = None
                    cur_modality_evidence_ls.append(
                        zero_task_evidence if cur_task_evidence is None else cur_task_evidence
                    )
                    if collect_task_evidence_maps:
                        cur_task_attn_maps = None if cur_task_evidence_aux is None else \
                            cur_task_evidence_aux.get('task_attn_maps', None)
                        cur_task_masks = None if cur_task_evidence_aux is None else \
                            cur_task_evidence_aux.get('task_masks', None)
                        cur_task_evidence_maps = None if cur_task_evidence_aux is None else \
                            cur_task_evidence_aux.get('task_evidence_maps', None)
                        if cur_task_attn_maps is not None:
                            template_task_attn_maps = cur_task_attn_maps.new_zeros(cur_task_attn_maps.shape)
                        if cur_task_masks is not None:
                            template_task_masks = cur_task_masks.new_zeros(cur_task_masks.shape)
                        cur_modality_attn_ls.append(cur_task_attn_maps)
                        cur_modality_mask_ls.append(cur_task_masks)
                        cur_modality_evidence_map_ls.append(cur_task_evidence_maps)
                else:
                    cur_expert_output = self.expert_ls[modality_idx](
                        x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...]
                    )
                cur_modality_output_ls.append(cur_expert_output)

            modality_output_ls.append(torch.cat(cur_modality_output_ls, dim=0))
            if collect_task_evidence:
                modality_evidence_ls.append(torch.cat(cur_modality_evidence_ls, dim=0))
            if collect_task_evidence_maps:
                modality_attn_map_ls.append(cur_modality_attn_ls)
                modality_target_mask_ls.append(cur_modality_mask_ls)
                modality_evidence_map_ls.append(cur_modality_evidence_map_ls)

        modality_feature = torch.stack(modality_output_ls, dim=1)  # [B, M, C, H, W, D]
        task_evidence = torch.stack(modality_evidence_ls, dim=1) if collect_task_evidence else None
        (
            router_weight,
            task_router_weight,
            router_logits,
            router_uncertainty,
            router_score,
            router_denom,
            task_router_logits,
            task_router_global_component,
            task_router_task_component,
            task_router_beta
        ) = self._compute_feature_router_weights(x, modality_feature, modality_mask, task_evidence=task_evidence)
        fused_feature = (
            modality_feature * router_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)

        if not collect_task_evidence:
            return self.segmentation_head(fused_feature)

        modality_reliability = None
        if router_uncertainty is not None:
            # TE-DMoE: convert uncertainty to confidence-like reliability in (0, 1].
            modality_reliability = torch.exp(-router_uncertainty.squeeze(-1)).detach()
        if self.runtime_stage1_teacher_mode:
            # TE-DMoE: stage-1 teacher training does not learn predictor yet.
            predicted_task_evidence = torch.zeros_like(task_evidence)
            modality_available = (~modality_mask).float().unsqueeze(-1)
            available_count = modality_available.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_task_evidence = (task_evidence * modality_available).sum(dim=1, keepdim=True) / available_count
            mean_task_evidence = mean_task_evidence.expand_as(task_evidence)
        else:
            predicted_task_evidence, mean_task_evidence = _predict_missing_task_evidence(
                self.missing_evidence_predictor,
                task_evidence,
                modality_mask,
                modality_reliability=modality_reliability
            )
        evidence_aux = _build_task_evidence_aux(
            task_evidence,
            modality_mask,
            predicted_task_evidence=predicted_task_evidence,
            mean_task_evidence=mean_task_evidence,
            missing_fusion_mean_weight=self.missing_evidence_mean_weight,
            missing_fusion_pred_weight=self.missing_evidence_pred_weight
        )
        if collect_task_evidence_maps:
            # TE-DMoE: materialize dense [B, M, T, h, w, d] tensors with zeros for missing slots.
            if template_task_attn_maps is None:
                down_factor = 2 ** max(0, self.n_stages - 1)
                hh = max(1, x.shape[2] // down_factor)
                ww = max(1, x.shape[3] // down_factor)
                dd = max(1, x.shape[4] // down_factor)
                template_task_attn_maps = x.new_zeros((1, self.num_evidence_tasks, hh, ww, dd))
                template_task_masks = x.new_zeros((1, self.num_evidence_tasks, hh, ww, dd))
            if template_task_masks is None:
                template_task_masks = x.new_zeros(template_task_attn_maps.shape)

            attn_stack_by_modality = []
            mask_stack_by_modality = []
            evidence_map_stack_by_modality = []
            for modality_idx in range(4):
                cur_attn_dense_ls = []
                cur_mask_dense_ls = []
                cur_evidence_map_dense_ls = []
                for sample_idx in range(x.shape[0]):
                    cur_attn = modality_attn_map_ls[modality_idx][sample_idx]
                    cur_mask = modality_target_mask_ls[modality_idx][sample_idx]
                    cur_evidence_map = modality_evidence_map_ls[modality_idx][sample_idx]
                    cur_attn_dense_ls.append(template_task_attn_maps if cur_attn is None else cur_attn)
                    cur_mask_dense_ls.append(template_task_masks if cur_mask is None else cur_mask)
                    if cur_evidence_map is None:
                        dim_per_task = self.task_evidence_dim // self.num_evidence_tasks
                        cur_evidence_map_dense_ls.append(
                            x.new_zeros(
                                (1, self.num_evidence_tasks, dim_per_task, template_task_attn_maps.shape[-3],
                                 template_task_attn_maps.shape[-2], template_task_attn_maps.shape[-1])
                            )
                        )
                    else:
                        cur_evidence_map_dense_ls.append(cur_evidence_map)
                attn_stack_by_modality.append(torch.cat(cur_attn_dense_ls, dim=0))
                mask_stack_by_modality.append(torch.cat(cur_mask_dense_ls, dim=0))
                evidence_map_stack_by_modality.append(torch.cat(cur_evidence_map_dense_ls, dim=0))

            evidence_aux['task_evidence_attn_maps'] = torch.stack(attn_stack_by_modality, dim=1)
            evidence_aux['task_evidence_target_masks'] = torch.stack(mask_stack_by_modality, dim=1)
            evidence_aux['task_evidence_maps'] = torch.stack(evidence_map_stack_by_modality, dim=1)
        decoder_aux = None
        z_comp = evidence_aux['completed_task_evidence']  # [B, M, E]
        if self.use_task_specific_router:
            # TE-DMoE: task-router bias branch uses completed evidence for stronger task conditioning.
            (
                _,
                task_router_weight,
                _,
                _,
                _,
                _,
                task_router_logits,
                task_router_global_component,
                task_router_task_component,
                task_router_beta
            ) = self._compute_feature_router_weights(x, modality_feature, modality_mask, task_evidence=z_comp)
        z_comp_split = z_comp.view(z_comp.shape[0], z_comp.shape[1], self.num_evidence_tasks, -1)  # [B, M, T, D]
        if task_router_weight is None:
            task_router_weight = router_weight.unsqueeze(-1).expand(-1, -1, self.num_evidence_tasks)
        task_evidence_summary = (z_comp_split * task_router_weight.unsqueeze(-1)).sum(dim=1)  # [B, T, D]
        z_cond = task_evidence_summary.reshape(task_evidence_summary.shape[0], -1)  # [B, E]
        plain_output = self.segmentation_head(fused_feature)
        hierarchical_output = None
        task_spatial_fusion_aux = None
        film_aux = None
        if (self.task_spatial_evidence_fusion is not None) and (not self.runtime_stage1_teacher_mode):
            # TE-DMoE: main branch: task-aware spatial gating + hierarchical composition.
            output, modulated_feature, task_spatial_fusion_aux = self.task_spatial_evidence_fusion(
                fused_feature, task_evidence_summary
            )
            hierarchical_output = output
            decoder_aux = task_spatial_fusion_aux
            if self.enable_legacy_evidence_output_residual:
                output = plain_output + float(self.evidence_hier_residual_scale) * output
        elif (self.evidence_film is not None) and (not self.runtime_stage1_teacher_mode):
            # TE-DMoE: main strong-coupling path; completed evidence modulates fused bottleneck feature.
            modulated_feature, film_aux = self.evidence_film(fused_feature, z_cond)
            output = self.segmentation_head(modulated_feature)
            if self.enable_legacy_evidence_output_residual:
                hierarchical_output, decoder_aux = self.task_conditioned_decoder(fused_feature, z_comp)
                output = output + float(self.evidence_hier_residual_scale) * hierarchical_output
        elif self.enable_legacy_evidence_output_residual and (not self.runtime_stage1_teacher_mode):
            # TE-DMoE: legacy compatibility path (default should be disabled in new configs).
            hierarchical_output, decoder_aux = self.task_conditioned_decoder(fused_feature, z_comp)
            output = plain_output + float(self.evidence_hier_residual_scale) * hierarchical_output
        else:
            output = plain_output

        final_output, evidence_residual_logits, evidence_residual_gamma, evidence_residual_beta, evidence_residual_scale = \
            _apply_task_evidence_residual(
                self.evidence_residual_fusion,
                output,
                evidence_aux,
                use_completed_evidence_for_residual=self.use_completed_evidence_for_residual
            )
        if not return_aux:
            return final_output

        task_prototypes = self.get_task_prototypes()
        return dict(
            logits=final_output,
            original_logits=output,
            evidence_residual_logits=evidence_residual_logits,
            evidence_residual_gamma=evidence_residual_gamma,
            evidence_residual_beta=evidence_residual_beta,
            evidence_residual_bias=evidence_residual_beta,
            evidence_residual_scale=evidence_residual_scale,
            plain_logits=plain_output,
            hierarchical_logits=hierarchical_output,
            hierarchical_residual_scale=float(self.evidence_hier_residual_scale),
            fused_feature=fused_feature,
            fused_feature_before_film=film_aux.get('fused_feature_before_film', fused_feature) if film_aux is not None else fused_feature,
            fused_feature_after_film=film_aux.get('fused_feature_after_film', fused_feature) if film_aux is not None else fused_feature,
            film_gamma=film_aux.get('film_gamma', None) if film_aux is not None else None,
            film_beta=film_aux.get('film_beta', None) if film_aux is not None else None,
            film_alpha_gamma=film_aux.get('film_alpha_gamma', None) if film_aux is not None else None,
            film_alpha_beta=film_aux.get('film_alpha_beta', None) if film_aux is not None else None,
            evidence_condition_vector=z_cond,
            task_evidence_summary=task_evidence_summary,
            bottleneck_spatial_attention=film_aux.get('bottleneck_spatial_attention', None) if film_aux is not None else None,
            task_spatial_gate_maps=task_spatial_fusion_aux.get('task_gate_maps', None) if task_spatial_fusion_aux is not None else None,
            gate_r1=task_spatial_fusion_aux.get('gate_r1', None) if task_spatial_fusion_aux is not None else None,
            gate_r2=task_spatial_fusion_aux.get('gate_r2', None) if task_spatial_fusion_aux is not None else None,
            gate_r3=task_spatial_fusion_aux.get('gate_r3', None) if task_spatial_fusion_aux is not None else None,
            delta_r1=task_spatial_fusion_aux.get('delta_r1', None) if task_spatial_fusion_aux is not None else None,
            delta_r2=task_spatial_fusion_aux.get('delta_r2', None) if task_spatial_fusion_aux is not None else None,
            delta_r3=task_spatial_fusion_aux.get('delta_r3', None) if task_spatial_fusion_aux is not None else None,
            alpha_r1=task_spatial_fusion_aux.get('alpha_r1', None) if task_spatial_fusion_aux is not None else None,
            alpha_r2=task_spatial_fusion_aux.get('alpha_r2', None) if task_spatial_fusion_aux is not None else None,
            alpha_r3=task_spatial_fusion_aux.get('alpha_r3', None) if task_spatial_fusion_aux is not None else None,
            task_spatial_region_logits=task_spatial_fusion_aux.get('region_logits', None) if task_spatial_fusion_aux is not None else None,
            task_spatial_alpha=task_spatial_fusion_aux.get('task_alpha', None) if task_spatial_fusion_aux is not None else None,
            evidence_guided_decoder_aux=decoder_aux,
            region_logits=decoder_aux.get('region_logits', None) if decoder_aux is not None else None,
            logits_r1=decoder_aux.get('logits_r1', None) if decoder_aux is not None else None,
            logits_r2=decoder_aux.get('logits_r2', None) if decoder_aux is not None else None,
            logits_r3=decoder_aux.get('logits_r3', None) if decoder_aux is not None else None,
            router_weight=router_weight,
            router_weights=router_weight,
            task_router_weight=task_router_weight,
            task_router_logits=task_router_logits,
            task_router_global_component=task_router_global_component,
            task_router_task_component=task_router_task_component,
            task_router_beta=task_router_beta,
            router_logits=router_logits,
            router_uncertainty=router_uncertainty,
            router_score=router_score,
            router_denom=router_denom,
            hierarchical_region_names=self.hierarchical_region_names,
            use_hierarchical_output_composition=self.use_hierarchical_output_composition,
            task_prototypes=task_prototypes,
            runtime_stage1_teacher_mode=self.runtime_stage1_teacher_mode,
            **evidence_aux
        )
