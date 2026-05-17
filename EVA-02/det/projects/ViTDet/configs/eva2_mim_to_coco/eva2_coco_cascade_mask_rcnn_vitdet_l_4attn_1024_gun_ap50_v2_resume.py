"""EVA-02 Large + ARMED: Gun AP50 v2 — RESUME-SAFE version.

Identical to gun_ap50_v2 except it KEEPS the original anchor sizes so that
100% of the checkpoint weights load without any shape mismatch.

This config is designed to be used with:
    train.init_checkpoint=output/model_XXXXX.pth
so the model starts from the accuracy you've already achieved
(Armed ~45, Unarmed ~41, Gun ~22) and improves from there.

Changes applied (all checkpoint-compatible):
 1. Per-class focal alpha:  Gun α=0.75, Armed/Unarmed α=0.25, BG α=0.10
 2. GIoU box regression loss (scale-invariant — helps tiny Gun boxes)
 3. Higher learning rate 5e-6 — escape the lr=1e-6 convergence plateau
 4. Stronger repeat-factor threshold (0.05) for Gun oversampling
 5. Low-quality cascade matching in stage 0 — every GT gun gets a proposal
 6. More aggressive scale augmentation (min_scale=0.5, max_scale=2.5)
 7. Lower soft-NMS IoU threshold (0.2) so overlapping Gun+Armed survive
 8. Extended schedule to 100k with longer warmup (stability at higher LR)

NOTE: Anchor sizes are UNCHANGED from the one-shot config to guarantee
      full checkpoint compatibility. See gun_ap50_v2.py for the version
      with multi-scale anchors (requires fresh start or accepts RPN re-init).
"""

from detectron2.config import LazyCall as L
import detectron2.data.transforms as T
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.modeling.matcher import Matcher

from .eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_lrd0p8_bs2_lr1e6 import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
)

# ---------------------------------------------------------------------------
# 1) Aggressive small-object augmentation — wider scale range than v1.
#    min_scale=0.5 zooms in more (makes small guns bigger in the crop);
#    max_scale=2.5 also helps by creating harder negatives at small scale.
# ---------------------------------------------------------------------------
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.RandomContrast)(intensity_min=0.6, intensity_max=1.4),
    L(T.RandomBrightness)(intensity_min=0.6, intensity_max=1.4),
    L(T.RandomSaturation)(intensity_min=0.5, intensity_max=1.5),
    L(T.ResizeScale)(
        min_scale=0.5,
        max_scale=2.5,
        target_height=1024,
        target_width=1024,
    ),
    L(T.FixedSizeCrop)(crop_size=(1024, 1024), pad=False),
]

# ---------------------------------------------------------------------------
# 2) Stronger class-imbalance mitigation: raise threshold 0.01 → 0.05.
#    With 3 classes this gives ~1.4-1.8× oversampling for Gun images.
# ---------------------------------------------------------------------------
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.05
    )
)

# ---------------------------------------------------------------------------
# 3) SAME anchor sizes as v1 (checkpoint-compatible).
#    Each level: 1 size × 3 aspect ratios = 3 anchors per cell.
#    This ensures the RPN head conv weights load perfectly from checkpoint.
# ---------------------------------------------------------------------------
model.proposal_generator.anchor_generator.sizes = [
    [16],
    [32],
    [64],
    [128],
    [256],
]

# ---------------------------------------------------------------------------
# 4) Cascade matchers: allow low-quality matches in stage 0 so every GT gun
#    is matched to at least one proposal (even if IoU is weak for tiny guns).
#    Stages 1-2 keep strict matching to refine.
# ---------------------------------------------------------------------------
model.roi_heads.proposal_matchers = [
    L(Matcher)(thresholds=[0.4], labels=[0, 1], allow_low_quality_matches=True),
    L(Matcher)(thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False),
    L(Matcher)(thresholds=[0.6], labels=[0, 1], allow_low_quality_matches=False),
]

# ---------------------------------------------------------------------------
# 5) Soft-NMS: lower IoU threshold (0.2 vs 0.3) so overlapping Gun + Armed
#    detections both survive. Guns are always inside/overlapping Armed boxes.
# ---------------------------------------------------------------------------
model.roi_heads.use_soft_nms = True
model.roi_heads.method = "linear"
model.roi_heads.iou_threshold = 0.2
model.roi_heads.sigma = 0.5
model.roi_heads.class_wise = True

# ---------------------------------------------------------------------------
# 6) RPN proposal capacity: same as v1 (already generous).
# ---------------------------------------------------------------------------
model.proposal_generator.pre_nms_topk = (4000, 2000)
model.proposal_generator.post_nms_topk = (2000, 2000)

# ---------------------------------------------------------------------------
# 7) ROI sampling: same 512 / 0.5 as v1.
# ---------------------------------------------------------------------------
model.roi_heads.batch_size_per_image = 512
model.roi_heads.positive_fraction = 0.5

# ---------------------------------------------------------------------------
# 8) Per-class focal loss alpha + GIoU box regression.
#
#    Alpha weights (indexed by class):
#      class 0 = Armed   → 0.25 (dominant, keep normal)
#      class 1 = Unarmed → 0.25 (dominant, keep normal)
#      class 2 = Gun     → 0.75 (3× boost for rare/hard class)
#      class 3 = BG      → 0.10 (de-weight easy background)
#
#    GIoU loss is scale-invariant: a 2px error on a 30px gun is penalized
#    proportionally the same as a 20px error on a 300px person.
#
#    NOTE: focal_loss_alpha as a list is stored as a buffer tensor, not in
#    the v1 checkpoint's state_dict (which stored it as a plain float).
#    This is safe — the buffer will be freshly initialized from this config.
# ---------------------------------------------------------------------------
for _bp in model.roi_heads.box_predictors:
    _bp.use_focal_loss = True
    _bp.focal_loss_alpha = [0.25, 0.25, 0.75, 0.10]
    _bp.focal_loss_gamma = 2.0
    _bp.box_reg_loss_type = "giou"

# ---------------------------------------------------------------------------
# 9) Higher learning rate: 5e-6 (5× the v1 rate of 1e-6).
#    The model plateaued at lr=1e-6 after ~24k iters; this gives enough
#    gradient signal to adjust Gun-specific features while preserving
#    Armed/Unarmed accuracy from the loaded checkpoint.
# ---------------------------------------------------------------------------
optimizer.lr = 5e-6

# ---------------------------------------------------------------------------
# 10) Schedule: 100k iters with milestones at 80k and 90k.
#     When resuming from iter ~54k, this gives ~46k more iters of training
#     at the higher LR + better loss function. Warmup 2000 iters.
# ---------------------------------------------------------------------------
train.max_iter = 100000
lr_multiplier.scheduler.milestones = [
    train.max_iter * 8 // 10,
    train.max_iter * 9 // 10,
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 2000 / train.max_iter
