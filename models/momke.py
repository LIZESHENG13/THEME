import torch
from torch import nn

from configs_joint_training import ModelConfig
from models.evidence_residual_fusion import EvidenceResidualFusion
from models.missing_evidence_predictor import MissingEvidencePredictor
from models.nnunet import UNet


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
    print(
        f"[TE-DMoE][partial_load:{module_name}] "
        f"loaded={len(filtered_state)}, missing={len(missing_keys)}, "
        f"unexpected={len(unexpected_keys)}, skip_prefix={len(skipped_by_prefix)}, "
        f"skip_shape={len(skipped_by_shape)}"
    )


def _get_expert_partial_skip_prefixes():
    return [
        'task_score_head.',
        'task_token_proj.',
        'task_token_agg.',
        'evidence_head.',
        'evidence_pool.',
    ]


# TE-DMoE: standardize optional evidence packaging for joint-training consumers.
def _build_task_evidence_aux(stacked_evidence, modality_mask, predicted_task_evidence=None):
    if predicted_task_evidence is None:
        predicted_task_evidence = torch.zeros_like(stacked_evidence)
    modality_available_mask = ~modality_mask
    completed_task_evidence = torch.where(
        modality_available_mask.unsqueeze(-1),
        stacked_evidence,
        predicted_task_evidence
    )
    return {
        'task_evidence': stacked_evidence,
        'predicted_task_evidence': predicted_task_evidence,
        'completed_task_evidence': completed_task_evidence,
        'task_evidence_list': [stacked_evidence[:, modality_idx, :] for modality_idx in range(stacked_evidence.shape[1])],
        'modality_mask': modality_mask,
        'modality_available_mask': modality_available_mask,
    }


# TE-DMoE: predict missing evidence from observed evidence only.
def _predict_missing_task_evidence(missing_evidence_predictor, observed_task_evidence, modality_mask):
    if missing_evidence_predictor is None:
        return torch.zeros_like(observed_task_evidence)

    modality_available_mask = ~modality_mask
    return missing_evidence_predictor(observed_task_evidence, modality_available_mask)


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


class ExpertNet(UNet):
    def __init__(self, input_channels, n_classes, n_stages, n_features_per_stage, kernel_size, strides,
                 use_task_evidence=False, evidence_dim_per_task=64, num_evidence_tasks=3, evidence_pooling='avg',
                 use_task_conditioned_evidence=True, task_evidence_mask_source='gt_then_pred',
                 task_evidence_use_gt_mask_in_train=True, task_evidence_use_pred_mask_in_eval=True,
                 evidence_num_tokens=2, use_hierarchical_evidence=True,
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

    def forward(self, x, return_evidence=False, evidence_labels=None):
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
            if self.use_task_evidence:
                seg_logits_for_mask = self.seg_layers[-1](output)
                evidence = self._extract_task_evidence(
                    encoded_feat_maps[-1],
                    seg_logits=seg_logits_for_mask,
                    evidence_labels=evidence_labels
                )
            else:
                evidence = None
            return output, evidence

        return output


# MoMKE implementation. The original paper does not evaluate on BraTS2018; hence we implemented it by ourselves.
class MoMKE(nn.Module):
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
        self.task_evidence_dim = getattr(ModelConfig, 'EVIDENCE_DIM_PER_TASK', 64) * \
            getattr(ModelConfig, 'NUM_EVIDENCE_TASKS', 3)
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
        self.router_ls = nn.ModuleList([
            self._build_router(),
            self._build_router(),
            self._build_router(),
            self._build_router()
        ])
        self.segmentation_head = nn.Conv3d(
            in_channels=4 * self.expert_ls[0].seg_layers[-1].in_channels,
            out_channels=self.expert_ls[0].seg_layers[-1].out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

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

        expert_state_dict = torch.load(self.pretrained_expert_file_list[expert_id])
        _load_partial_state_dict_with_report(
            net,
            expert_state_dict,
            module_name=f'MoMKE.expert_{expert_id}',
            skip_prefixes=_get_expert_partial_skip_prefixes()
        )

        return net

    def _build_router(self):
        router = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),
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

        return router

    def forward(self, x, return_aux=False):
        modality_mask = (x == 0).all(dim=-1).all(dim=-1).all(dim=-1)

        collect_task_evidence = self.use_task_evidence and (return_aux or self.enable_evidence_residual_fusion)
        if not collect_task_evidence:
            output = torch.cat([
                torch.cat([
                    torch.zeros(self.expert_output_shape).cuda()
                    if modality_mask[sample_idx, modality_idx]
                    else
                    (
                            torch.stack([self.expert_ls[expert_idx](
                                x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...])
                                         for expert_idx in range(4)], dim=1) *
                            (
                                nn.functional.softmax(
                                    self.router_ls[modality_idx](
                                        x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...]), dim=1
                                ).view(-1, 4, 1, 1, 1, 1)
                            )
                    ).sum(dim=1)

                    for sample_idx in range(x.shape[0])
                ], dim=0)

                for modality_idx in range(4)
            ], dim=1)

            output = self.segmentation_head(output)

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

                router_weight = nn.functional.softmax(
                    self.router_ls[modality_idx](
                        x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...]), dim=1
                ).view(-1, 4, 1, 1, 1, 1)
                stacked_expert_outputs = []
                stacked_expert_evidence = []
                for expert_idx in range(4):
                    cur_expert_output, cur_task_evidence = self.expert_ls[expert_idx](
                        x[sample_idx:sample_idx + 1, modality_idx:modality_idx + 1, ...], return_evidence=True
                    )
                    stacked_expert_outputs.append(cur_expert_output)
                    stacked_expert_evidence.append(
                        zero_task_evidence if cur_task_evidence is None else cur_task_evidence
                    )

                stacked_expert_outputs = torch.stack(stacked_expert_outputs, dim=1)
                stacked_expert_evidence = torch.stack(stacked_expert_evidence, dim=1)
                fused_output = (stacked_expert_outputs * router_weight).sum(dim=1)
                fused_task_evidence = (
                    stacked_expert_evidence * router_weight.view(-1, 4, 1)
                ).sum(dim=1)
                cur_modality_output_ls.append(fused_output)
                cur_modality_evidence_ls.append(fused_task_evidence)

            modality_output_ls.append(torch.cat(cur_modality_output_ls, dim=0))
            modality_evidence_ls.append(torch.cat(cur_modality_evidence_ls, dim=0))

        output = torch.cat(modality_output_ls, dim=1)
        output = self.segmentation_head(output)

        task_evidence = torch.stack(modality_evidence_ls, dim=1)
        predicted_task_evidence = _predict_missing_task_evidence(
            self.missing_evidence_predictor, task_evidence, modality_mask
        )
        evidence_aux = _build_task_evidence_aux(task_evidence, modality_mask, predicted_task_evidence)
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
