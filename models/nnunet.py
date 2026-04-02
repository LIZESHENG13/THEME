import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary


# This 3D-UNet structure is from nnUNet's configuration for BraTS2018 task. Thanks, nnUNet authors!
class UNet(nn.Module):
    def __init__(self, input_channels,
                 n_classes, n_stages, n_features_per_stage, kernel_size, strides,
                 apply_deep_supervision=False,
                 use_task_evidence=False,
                 evidence_dim_per_task=64,
                 num_evidence_tasks=3,
                 evidence_pooling='avg',
                 use_task_conditioned_evidence=True,
                 task_evidence_mask_source='gt_then_pred',
                 task_evidence_use_gt_mask_in_train=True,
                 task_evidence_use_pred_mask_in_eval=True,
                 evidence_num_tokens=2,
                 use_hierarchical_evidence=True,
                 hierarchical_region_names=("R1_ET", "R2_TC_MINUS_ET", "R3_WT_MINUS_TC")):
        super().__init__()

        self.input_channels = input_channels
        self.n_classes = n_classes
        self.n_stages = n_stages
        self.n_features_per_stage = n_features_per_stage
        self.kernel_size = kernel_size
        self.strides = strides
        self.apply_deep_supervision = apply_deep_supervision
        # TE-DMoE: optional per-expert task-evidence configuration.
        self.use_task_evidence = use_task_evidence
        self.evidence_dim_per_task = evidence_dim_per_task
        self.num_evidence_tasks = num_evidence_tasks
        self.evidence_pooling = evidence_pooling
        self.use_task_conditioned_evidence = use_task_conditioned_evidence
        self.task_evidence_mask_source = task_evidence_mask_source
        self.task_evidence_use_gt_mask_in_train = task_evidence_use_gt_mask_in_train
        self.task_evidence_use_pred_mask_in_eval = task_evidence_use_pred_mask_in_eval
        self.evidence_num_tokens = evidence_num_tokens
        self.use_hierarchical_evidence = use_hierarchical_evidence
        self.hierarchical_region_names = hierarchical_region_names
        self.total_evidence_dim = self.evidence_dim_per_task * self.num_evidence_tasks

        self._build_encoder()
        self._build_decoder()
        # TE-DMoE: build evidence branch only when explicitly enabled.
        if self.use_task_evidence:
            self._build_task_evidence_head()

    def _build_encoder(self):
        stages = []

        for i in range(self.n_stages):
            cur_stage = [
                nn.Conv3d(
                    in_channels=self.input_channels if i == 0 else self.n_features_per_stage[i - 1],
                    out_channels=self.n_features_per_stage[i],
                    kernel_size=self.kernel_size[i],
                    stride=self.strides[i],
                    padding=[(_ - 1) // 2 for _ in self.kernel_size[i]],
                    dilation=1,
                    bias=True,
                ),
                nn.modules.instancenorm.InstanceNorm3d(self.n_features_per_stage[i], eps=1e-05, affine=True),
                nn.modules.activation.LeakyReLU(inplace=True),

                nn.Conv3d(
                    in_channels=self.n_features_per_stage[i],
                    out_channels=self.n_features_per_stage[i],
                    kernel_size=self.kernel_size[i],
                    stride=1,
                    padding=[(_ - 1) // 2 for _ in self.kernel_size[i]],
                    dilation=1,
                    bias=True,
                ),
                nn.modules.instancenorm.InstanceNorm3d(self.n_features_per_stage[i], eps=1e-05, affine=True),
                nn.modules.activation.LeakyReLU(inplace=True),
            ]

            stages.append(nn.Sequential(*cur_stage))

        self.encoder_stages = nn.ModuleList(stages)

    def _build_decoder(self):
        connect_layers = []
        stages = []
        seg_layers = []

        for i in range(1, self.n_stages):
            connect_layers.append(
                nn.ConvTranspose3d(
                    in_channels=self.n_features_per_stage[-i],
                    out_channels=self.n_features_per_stage[-i - 1],
                    kernel_size=self.strides[-i],
                    stride=self.strides[-i],
                    bias=True
                )
            )

            cur_stage = [
                nn.ConvTranspose3d(
                    in_channels=2 * self.n_features_per_stage[-i - 1],
                    out_channels=self.n_features_per_stage[-i - 1],
                    kernel_size=self.kernel_size[-i - 1],
                    stride=1,
                    padding=[(_ - 1) // 2 for _ in self.kernel_size[-i - 1]],
                    dilation=1,
                    bias=True,
                ),
                nn.modules.instancenorm.InstanceNorm3d(self.n_features_per_stage[-i - 1], eps=1e-05, affine=True),
                nn.modules.activation.LeakyReLU(inplace=True),

                nn.ConvTranspose3d(
                    in_channels=self.n_features_per_stage[-i - 1],
                    out_channels=self.n_features_per_stage[-i - 1],
                    kernel_size=self.kernel_size[-i - 1],
                    stride=1,
                    padding=[(_ - 1) // 2 for _ in self.kernel_size[-i - 1]],
                    dilation=1,
                    bias=True,
                ),
                nn.modules.instancenorm.InstanceNorm3d(self.n_features_per_stage[-i - 1], eps=1e-05, affine=True),
                nn.modules.activation.LeakyReLU(inplace=True),
            ]

            stages.append(nn.Sequential(*cur_stage))

            seg_layers.append(
                nn.Conv3d(
                    in_channels=self.n_features_per_stage[-i - 1],
                    out_channels=self.n_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )

        self.connect_layers = nn.ModuleList(connect_layers)
        self.decoder_stages = nn.ModuleList(stages)
        self.seg_layers = nn.ModuleList(seg_layers)

    # TE-DMoE: project deepest encoder feature to task-evidence vector(s).
    def _build_task_evidence_head(self):
        if self.evidence_pooling != 'avg':
            raise ValueError(f'Unsupported evidence pooling: {self.evidence_pooling}. Expected "avg".')

        feat_dim = self.n_features_per_stage[-1]
        hidden_dim = max(32, feat_dim // 2)
        if not self.use_task_conditioned_evidence:
            self.evidence_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.evidence_head = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(hidden_dim, self.total_evidence_dim),
            )
            return

        # TE-DMoE: independent regional token extraction with mask-guided attention pooling.
        if self.num_evidence_tasks != 3:
            raise ValueError(
                f'num_evidence_tasks={self.num_evidence_tasks} is not supported; expected 3 for BraTS hierarchy.'
            )
        if self.use_hierarchical_evidence:
            self.task_name_ls = ['r1', 'r2', 'r3']
        else:
            self.task_name_ls = ['et', 'tc', 'wt']
        self.task_score_head = nn.ModuleDict({
            task_name: nn.Conv3d(feat_dim, self.evidence_num_tokens, kernel_size=1, stride=1, padding=0, bias=True)
            for task_name in self.task_name_ls
        })
        # TE-DMoE: task prototypes are query anchors, not final evidence outputs.
        self.task_prototypes = nn.ParameterDict({
            task_name: nn.Parameter(torch.randn(self.evidence_dim_per_task) * 0.02)
            for task_name in self.task_name_ls
        })
        self.task_key_proj = nn.ModuleDict({
            task_name: nn.Conv3d(feat_dim, self.evidence_dim_per_task, kernel_size=1, stride=1, padding=0, bias=True)
            for task_name in self.task_name_ls
        })
        self.task_value_proj = nn.ModuleDict({
            task_name: nn.Conv3d(feat_dim, self.evidence_dim_per_task, kernel_size=1, stride=1, padding=0, bias=True)
            for task_name in self.task_name_ls
        })
        self.task_token_proj = nn.ModuleDict({
            task_name: nn.Linear(feat_dim, self.evidence_dim_per_task)
            for task_name in self.task_name_ls
        })
        self.task_token_agg = nn.ModuleDict({
            task_name: nn.Sequential(
                nn.LayerNorm(self.evidence_num_tokens * self.evidence_dim_per_task),
                nn.Linear(self.evidence_num_tokens * self.evidence_dim_per_task, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.evidence_dim_per_task)
            )
            for task_name in self.task_name_ls
        })

    # TE-DMoE: resize task masks to deepest feature map resolution.
    @staticmethod
    def _resize_task_mask(task_mask, target_shape):
        if task_mask.shape[2:] == target_shape:
            return task_mask
        return F.interpolate(task_mask, size=target_shape, mode='trilinear', align_corners=False)

    # TE-DMoE: reorder channels from [WT, TC, ET] to [ET, TC, WT].
    @staticmethod
    def _to_et_tc_wt(task_tensor):
        return task_tensor[:, 2:3, ...], task_tensor[:, 1:2, ...], task_tensor[:, 0:1, ...]

    # TE-DMoE: build exclusive hierarchical masks from [WT, TC, ET].
    @staticmethod
    def _to_hierarchical_regions(wt_mask, tc_mask, et_mask):
        r1 = et_mask
        r2 = torch.clamp(tc_mask - et_mask, min=0.0, max=1.0)
        r3 = torch.clamp(wt_mask - tc_mask, min=0.0, max=1.0)
        return r1, r2, r3

    # TE-DMoE: build task masks from predicted probabilities.
    def _task_masks_from_logits(self, seg_logits, target_shape):
        probs = torch.sigmoid(seg_logits).detach()
        et_mask, tc_mask, wt_mask = self._to_et_tc_wt(probs)
        if self.use_hierarchical_evidence:
            r1, r2, r3 = self._to_hierarchical_regions(wt_mask, tc_mask, et_mask)
            return {
                'r1': self._resize_task_mask(r1, target_shape),
                'r2': self._resize_task_mask(r2, target_shape),
                'r3': self._resize_task_mask(r3, target_shape),
            }
        return {
            'et': self._resize_task_mask(et_mask, target_shape),
            'tc': self._resize_task_mask(tc_mask, target_shape),
            'wt': self._resize_task_mask(wt_mask, target_shape),
        }

    # TE-DMoE: build task masks from GT labels [B,3,H,W,D], channel order [WT,TC,ET].
    def _task_masks_from_gt(self, evidence_labels, target_shape):
        gt = evidence_labels.float()
        et_mask, tc_mask, wt_mask = self._to_et_tc_wt(gt)
        if self.use_hierarchical_evidence:
            r1, r2, r3 = self._to_hierarchical_regions(wt_mask, tc_mask, et_mask)
            return {
                'r1': self._resize_task_mask(r1, target_shape),
                'r2': self._resize_task_mask(r2, target_shape),
                'r3': self._resize_task_mask(r3, target_shape),
            }
        return {
            'et': self._resize_task_mask(et_mask, target_shape),
            'tc': self._resize_task_mask(tc_mask, target_shape),
            'wt': self._resize_task_mask(wt_mask, target_shape),
        }

    # TE-DMoE: choose GT/pred masks using train/eval policy.
    def _resolve_task_masks(self, seg_logits, evidence_labels, target_shape):
        mask_source = str(self.task_evidence_mask_source).lower()
        use_gt = False
        if self.training:
            if self.task_evidence_use_gt_mask_in_train and evidence_labels is not None:
                use_gt = mask_source in ['gt', 'gt_then_pred', 'gt_or_pred', 'auto']
        else:
            if (not self.task_evidence_use_pred_mask_in_eval) and evidence_labels is not None:
                use_gt = mask_source in ['gt', 'gt_then_pred', 'gt_or_pred', 'auto']

        if use_gt and evidence_labels is not None:
            return self._task_masks_from_gt(evidence_labels, target_shape)
        return self._task_masks_from_logits(seg_logits, target_shape)

    # TE-DMoE: mask-guided attention pooling to extract K regional tokens for one task.
    def _extract_task_tokens(self, deepest_feature, task_mask, task_name, eps=1e-6):
        # deepest_feature: [B, C, h, w, d], task_mask: [B, 1, h, w, d]
        raw_score = self.task_score_head[task_name](deepest_feature)  # [B, K, h, w, d]
        # TE-DMoE: prototype-conditioned attention query term.
        key_map = self.task_key_proj[task_name](deepest_feature)  # [B, D, h, w, d]
        task_proto = self.task_prototypes[task_name].view(1, self.evidence_dim_per_task, 1, 1, 1)
        proto_score = (key_map * task_proto).sum(dim=1, keepdim=True) / (self.evidence_dim_per_task ** 0.5)
        raw_score = raw_score + proto_score.expand_as(raw_score)
        safe_mask = task_mask.clamp_min(eps)
        mask_bias = torch.log(safe_mask).expand_as(raw_score)
        attn_logits = raw_score + mask_bias

        bsz, num_tokens, hh, ww, dd = attn_logits.shape
        attn = torch.softmax(attn_logits.view(bsz, num_tokens, -1), dim=-1).view(bsz, num_tokens, hh, ww, dd)
        tokens = torch.einsum('bkxyz,bcxyz->bkc', attn, deepest_feature)  # [B, K, C]
        tokens_proj = self.task_token_proj[task_name](tokens.reshape(bsz * num_tokens, -1))
        tokens_proj = tokens_proj.view(bsz, num_tokens, self.evidence_dim_per_task)  # [B, K, D]
        task_evidence = self.task_token_agg[task_name](tokens_proj.reshape(bsz, -1))  # [B, D]
        # TE-DMoE: explicit spatial evidence map for task-conditioned supervision.
        value_map = self.task_value_proj[task_name](deepest_feature)  # [B, D, h, w, d]
        attn_map = attn.mean(dim=1, keepdim=True)  # [B, 1, h, w, d]
        task_evidence_map = value_map * attn_map
        return task_evidence, tokens_proj, attn, task_evidence_map

    # TE-DMoE: extract evidence as concatenated region/task vectors built from K regional tokens each.
    def _extract_task_evidence(self, deepest_feature, seg_logits=None, evidence_labels=None, return_aux=False):
        if not self.use_task_conditioned_evidence:
            pooled_feature = self.evidence_pool(deepest_feature)
            flattened_feature = torch.flatten(pooled_feature, start_dim=1)
            evidence = self.evidence_head(flattened_feature)
            if not return_aux:
                return evidence
            return evidence, {
                'task_chunks': None,
                'task_tokens': None,
                'task_attn_maps': None,
                'task_evidence_maps': None,
                'task_masks': None,
                'task_names': list(getattr(self, 'task_name_ls', []))
            }

        if seg_logits is None:
            raise ValueError('seg_logits is required for task-conditioned evidence extraction.')

        task_masks = self._resolve_task_masks(seg_logits, evidence_labels, deepest_feature.shape[2:])
        evidence_ls = []
        task_chunk_ls = []
        task_token_ls = []
        task_attn_ls = []
        task_evidence_map_ls = []
        task_mask_ls = []
        for task_name in self.task_name_ls:
            cur_evidence, cur_tokens, cur_attn, cur_evidence_map = self._extract_task_tokens(
                deepest_feature, task_masks[task_name], task_name
            )
            evidence_ls.append(cur_evidence)
            if return_aux:
                task_chunk_ls.append(cur_evidence.unsqueeze(1))              # [B, 1, D]
                task_token_ls.append(cur_tokens.unsqueeze(1))               # [B, 1, K, D]
                # TE-DMoE: aggregate token-wise attention into one task map for alignment supervision.
                task_attn_ls.append(cur_attn.mean(dim=1, keepdim=True))     # [B, 1, h, w, d]
                task_evidence_map_ls.append(cur_evidence_map.unsqueeze(1))  # [B, 1, D, h, w, d]
                task_mask_ls.append(task_masks[task_name])                  # [B, 1, h, w, d]

        evidence = torch.cat(evidence_ls, dim=1)
        if not return_aux:
            return evidence

        evidence_aux = {
            'task_chunks': torch.cat(task_chunk_ls, dim=1),                 # [B, T, D]
            'task_tokens': torch.cat(task_token_ls, dim=1),                 # [B, T, K, D]
            'task_attn_maps': torch.cat(task_attn_ls, dim=1),               # [B, T, h, w, d]
            'task_evidence_maps': torch.cat(task_evidence_map_ls, dim=1),   # [B, T, D, h, w, d]
            'task_masks': torch.cat(task_mask_ls, dim=1),                   # [B, T, h, w, d]
            'task_names': list(self.task_name_ls)
        }
        return evidence, evidence_aux

    def forward(self, x, return_evidence=False, evidence_labels=None, return_evidence_aux=False):
        # TE-DMoE: evidence_labels follows training label order [WT, TC, ET].
        encoded_feat_maps = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoded_feat_maps.append(x)

        low_res_input = encoded_feat_maps[-1]
        outputs = []
        for i in range(len(self.decoder_stages)):
            output = self.connect_layers[i](low_res_input)
            output = torch.cat((output, encoded_feat_maps[-i - 2]), dim=1)
            output = self.decoder_stages[i](output)
            low_res_input = output

            output = self.seg_layers[i](output)

            if self.apply_deep_supervision:
                outputs.append(output)
            elif i == len(self.decoder_stages) - 1:
                outputs.append(output)

        if self.apply_deep_supervision:
            outputs = outputs[::-1]
        else:
            outputs = outputs[0]

        # TE-DMoE: return evidence only when explicitly requested.
        if return_evidence:
            evidence_aux = None
            if self.use_task_evidence:
                seg_logits = outputs[0] if isinstance(outputs, list) else outputs
                if return_evidence_aux:
                    evidence, evidence_aux = self._extract_task_evidence(
                        encoded_feat_maps[-1],
                        seg_logits=seg_logits,
                        evidence_labels=evidence_labels,
                        return_aux=True
                    )
                else:
                    evidence = self._extract_task_evidence(
                        encoded_feat_maps[-1],
                        seg_logits=seg_logits,
                        evidence_labels=evidence_labels
                    )
            else:
                evidence = None
            if return_evidence_aux:
                return outputs, evidence, evidence_aux
            return outputs, evidence

        return outputs


if __name__ == '__main__':
    net = UNet(
        input_channels=4,
        n_classes=3,
        n_stages=6,
        n_features_per_stage=[32, 64, 128, 256, 320, 320],
        kernel_size=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1], * [[2, 2, 2]] * 5],
        apply_deep_supervision=False
    ).cuda()
    summary(net, (4, 128, 128, 128))
    print(f'Output shape give input shape (2, 4, 128, 128, 128): {net(torch.rand(2, 4, 128, 128, 128).cuda()).shape}')
