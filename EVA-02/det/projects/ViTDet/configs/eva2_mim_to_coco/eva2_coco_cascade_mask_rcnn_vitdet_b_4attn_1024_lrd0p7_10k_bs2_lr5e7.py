from functools import partial
from pathlib import Path
import importlib.util
import sys

from projects.ViTDet.configs.common.coco_loader_lsj_1024 import dataloader

# Ensure dataset registration is available without relying on package imports.
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

# Pretrained model: EVA02 BSL on COCO
train.init_checkpoint = ""

# Dataset: ARMED (COCO-format custom)
dataloader.train.dataset.names = "armed_train"
dataloader.test.dataset.names = "armed_val"
dataloader.evaluator.dataset_name = "armed_val"
model.roi_heads.num_classes = 3

# Image size: 1024 (same as base config)
model.backbone.net.img_size = 1024
model.backbone.square_pad = 1024
model.backbone.net.patch_size = 16
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 768
model.backbone.net.depth = 12
model.backbone.net.num_heads = 12
model.backbone.net.mlp_ratio = 4 * 2 / 3
model.backbone.net.use_act_checkpoint = False
model.backbone.net.drop_path_rate = 0.1

# 2, 5, 8, 11 for global attention
model.backbone.net.window_block_indexes = [0, 1, 3, 4, 6, 7, 9, 10]

# Optimizer: AdamW, Base LR: 1e-6 (tuned up from 5e-7 for SPD conv layers)
optimizer.lr = 1e-6
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.7, num_layers=12)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

# Steps: 10000
train.max_iter = 50000
train.init_checkpoint=  "/kaggle/working/EVA/EVA-02/det/model_0008999.pth"
lr_multiplier.scheduler.milestones = [
    train.max_iter * 8 // 10,
    train.max_iter * 9 // 10,
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 1000 / train.max_iter

# Batch Size: 2
dataloader.test.num_workers = 0
dataloader.train.total_batch_size = 2


dataloader.train.mapper.use_instance_mask=False
dataloader.train.mapper.recompute_boxes=False

# Disable mask head for bbox-only training (dataset has no masks).
model.roi_heads.mask_in_features = None
model.roi_heads.mask_pooler = None
model.roi_heads.mask_head = None

# Enable Model EMA for smoother weights and better generalization.
train.model_ema.enabled = True
train.model_ema.decay = 0.9999
train.model_ema.use_ema_weights_for_eval_only = True

# Enable Soft-NMS for all cascade stages (reduces missed overlapping detections).
for _bp in model.roi_heads.box_predictors:
    _bp.use_soft_nms = True
    _bp.soft_nms_method = "linear"
    _bp.soft_nms_sigma = 0.5
    _bp.soft_nms_iou_threshold = 0.3

