from detectron2.config import LazyCall as L
import detectron2.data.transforms as T
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.modeling.matcher import Matcher

from .eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7_10k_bs2_lr5e7 import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
)

# One-shot AP50 experiment: prioritize small-object visibility and gun recall.

# 1) Small-object focused augmentation policy.
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.RandomContrast)(intensity_min=0.6, intensity_max=1.4),
    L(T.RandomBrightness)(intensity_min=0.6, intensity_max=1.4),
    L(T.RandomSaturation)(intensity_min=0.5, intensity_max=1.5),
    L(T.ResizeScale)(
        min_scale=0.8,
        max_scale=2.0,
        target_height=1024,
        target_width=1024,
    ),
    L(T.FixedSizeCrop)(crop_size=(1024, 1024), pad=False),
]

# 2) Class-imbalance mitigation for rare gun instances.
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.01
    )
)

# 3) Smaller RPN anchors (image-space) for tiny objects (e.g. guns) at p2–p6.
#    Keeps one size per level so num_anchors=3 (× aspect ratios) is unchanged.
model.proposal_generator.anchor_generator.sizes = [
    [16],
    [32],
    [64],
    [128],
    [256],
]

# 4) Revert cascade matchers to canonical thresholds so noisy small-gun
#    proposals still qualify as positives in stage 1 / stage 2.
model.roi_heads.proposal_matchers = [
    L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
    for th in [0.4, 0.5, 0.6]
]

# 5) Cascade-level Soft-NMS (CascadeROIHeads.forward consults these flags,
#    not the per-predictor flags). class_wise=True prevents adjacent Armed
#    and Gun detections from suppressing each other.
model.roi_heads.use_soft_nms = True
model.roi_heads.method = "linear"
model.roi_heads.iou_threshold = 0.3
model.roi_heads.sigma = 0.5
model.roi_heads.class_wise = True

# 6) RPN proposal capacity: more small-object proposals reach the cascade.
model.proposal_generator.pre_nms_topk = (4000, 2000)
model.proposal_generator.post_nms_topk = (2000, 2000)

# 7) ROI sampling: more positive RoIs per image so rare classes (Gun) get
#    enough supervision in each step.
model.roi_heads.batch_size_per_image = 512
model.roi_heads.positive_fraction = 0.5

# 8) Focal classification loss for all three cascade predictors. Down-weights
#    easy Armed/Unarmed examples so Gun gets stronger gradient.
for _bp in model.roi_heads.box_predictors:
    _bp.use_focal_loss = True
    _bp.focal_loss_alpha = 0.25
    _bp.focal_loss_gamma = 2.0

# 9) Moderate schedule extension for convergence.
train.max_iter = 80000
lr_multiplier.scheduler.milestones = [
    train.max_iter * 8 // 10,
    train.max_iter * 9 // 10,
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 1000 / train.max_iter
