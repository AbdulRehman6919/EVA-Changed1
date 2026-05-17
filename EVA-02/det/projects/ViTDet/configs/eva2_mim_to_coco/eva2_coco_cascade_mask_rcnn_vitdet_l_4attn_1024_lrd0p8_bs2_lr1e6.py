"""EVA-02 Large + ARMED dataset base config.

This is the L-variant analog of
`eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7_10k_bs2_lr5e7.py`.
It inherits the official L backbone (24 layers, embed_dim=1024, num_heads=16,
4 global-attn windows at indices 5/11/17/23, drop_path_rate=0.4, lrd=0.8) and
applies the SAME dataset / training conventions used in the B base config:

  * ARMED COCO-format dataset (3 classes: Armed, Unarmed, Gun).
  * BBox-only training (no mask head; ARMED has no mask annotations).
  * Image size 1024, batch size 2 (Kaggle 2x16GB regime).
  * LR 1e-6 with warmup 1000/max_iter (same conservative LR as the B run).
  * Activation checkpointing ON (L = 304M params; required to fit at bs=2 on T4).
  * Model EMA enabled (decay 0.9999, eval-only).
  * Per-stage Soft-NMS flags kept on the box predictors for legacy symmetry
    (CascadeROIHeads-level Soft-NMS is set in the one-shot config that imports
    this file).

ARCHITECTURAL changes from your B run (SPD in SimpleFeaturePyramid/LastLevelSPD,
focal-loss option in FastRCNNOutputLayers) live in shared code (`vit.py`,
`fast_rcnn.py`) and apply here automatically; no extra wiring needed.
"""

from functools import partial
from pathlib import Path
import importlib.util

from projects.ViTDet.configs.common.coco_loader_lsj_1024 import dataloader

# Locate datasets/register_armed.py without relying on package imports
# (matches the boilerplate in the B base config).
_register_path = None
for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "datasets" / "register_armed.py"
    if _candidate.exists():
        _register_path = _candidate
        break
if _register_path is None:
    raise FileNotFoundError("Missing dataset registration at datasets/register_armed.py")
_spec = importlib.util.spec_from_file_location("register_armed", _register_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

from .cascade_mask_rcnn_vitdet_b_100ep import (
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

# ---------------------------------------------------------------------------
# Backbone: EVA-02 Large (override the inherited B backbone)
# ---------------------------------------------------------------------------
model.backbone.net.img_size = 1024
model.backbone.square_pad = 1024
model.backbone.net.patch_size = 16
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 4 * 2 / 3
# Activation checkpointing ON for L (304M params won't fit at bs=2 on T4 without it).
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.4

# 4 global-attn windows at indices 5, 11, 17, 23 (canonical L schedule).
model.backbone.net.window_block_indexes = (
    list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
)

# ---------------------------------------------------------------------------
# Dataset: ARMED (COCO format, 3 classes)
# ---------------------------------------------------------------------------
dataloader.train.dataset.names = "armed_train"
dataloader.test.dataset.names = "armed_val"
dataloader.evaluator.dataset_name = "armed_val"
model.roi_heads.num_classes = 3

# Bbox-only training: ARMED has no instance masks.
dataloader.train.mapper.use_instance_mask = False
dataloader.train.mapper.recompute_boxes = False
model.roi_heads.mask_in_features = None
model.roi_heads.mask_pooler = None
model.roi_heads.mask_head = None

# ---------------------------------------------------------------------------
# Optimizer: AdamW, conservative LR 1e-6 at total_batch_size=2.
# Layer-wise LR decay 0.8 (24 layers) so early layers see ~LR * 0.8^24 ~= 0.005.
# ---------------------------------------------------------------------------
optimizer.lr = 1e-6
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24
)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

# ---------------------------------------------------------------------------
# Schedule: 50k iters as the base (one-shot config can extend to 80k).
# Overwrite this value in the one-shot config; just leave a sane default here.
# ---------------------------------------------------------------------------
train.max_iter = 50000
lr_multiplier.scheduler.milestones = [
    train.max_iter * 8 // 10,
    train.max_iter * 9 // 10,
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 1000 / train.max_iter

# Init checkpoint placeholder. Overwrite with the EVA-02 L pretrained weights
# (e.g. eva02_L_coco_bsl.pth, ~1.2GB) on the command line:
#   train.init_checkpoint=/kaggle/working/eva02_L_coco_bsl.pth
train.init_checkpoint = ""

# ---------------------------------------------------------------------------
# Batching: 2 (Kaggle regime). num_workers=0 to avoid worker-spawn issues.
# ---------------------------------------------------------------------------
dataloader.train.total_batch_size = 2
dataloader.test.num_workers = 0

# ---------------------------------------------------------------------------
# Model EMA: smoothed weights, eval-only.
# ---------------------------------------------------------------------------
train.model_ema.enabled = True
train.model_ema.decay = 0.9999
train.model_ema.use_ema_weights_for_eval_only = True

# ---------------------------------------------------------------------------
# Legacy per-stage Soft-NMS flags (kept for parity with B base; the actual
# inference Soft-NMS gate lives on CascadeROIHeads and is set in the
# one-shot config below).
# ---------------------------------------------------------------------------
for _bp in model.roi_heads.box_predictors:
    _bp.use_soft_nms = True
    _bp.soft_nms_method = "linear"
    _bp.soft_nms_sigma = 0.5
    _bp.soft_nms_iou_threshold = 0.3
