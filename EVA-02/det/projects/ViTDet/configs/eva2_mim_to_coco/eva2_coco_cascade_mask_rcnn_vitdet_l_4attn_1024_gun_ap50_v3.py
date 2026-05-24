"""EVA-02 Large + ARMED: Gun AP50 v3 — Cosine LR + Stronger Gun Focus.

This config is designed to BREAK the ~24% Gun AP plateau.

KEY DIFFERENCE from v2_resume:
    Do NOT use --resume! Use train.init_checkpoint to load just the model
    weights. This creates a FRESH optimizer and LR scheduler.

Launch command:
    python tools/lazyconfig_train_net.py \
        --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v3.py \
        --num-gpus 2 \
        train.init_checkpoint=output/model_0071999.pth \
        train.output_dir=output_v3 \
        train.eval_period=2000 \
        train.checkpointer.period=2000

Changes over v2_resume:
 1. COSINE LR schedule at base LR=1e-5 (vs multi-step at 5e-6 that never
    took effect because --resume carried the old optimizer state at 1e-6).
    Cosine keeps LR high for much longer → more gradient for Gun.
 2. Stronger focal loss: Gun alpha=0.85, gamma=3.0 (was 0.75/2.0).
 3. Higher repeat-factor threshold: 0.10 (was 0.05) → more Gun oversampling.
 4. Low-quality matching on ALL 3 cascade stages (was only stage 0).
    Every GT Gun box gets a positive proposal even in later stages.
 5. Higher positive_fraction: 0.75 (was 0.5) → each ROI batch has more
    positives, which means more Gun supervision per step.
 6. Copy-paste augmentation: pastes Gun instance crops onto each training
    image with 50% probability, adding 1-3 extra Gun instances per image.
"""

from detectron2.config import LazyCall as L
import detectron2.data.transforms as T
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.modeling.matcher import Matcher
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import CosineParamScheduler

from .eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_lrd0p8_bs2_lr1e6 import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
)

# ---------------------------------------------------------------------------
# 1) Aggressive small-object augmentation — same as v2_resume.
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
# 2) Stronger class-imbalance mitigation: repeat_thresh 0.05 → 0.10.
#    This gives ~2-3× oversampling for Gun images instead of ~1.4-1.8×.
# ---------------------------------------------------------------------------
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.10
    )
)

# ---------------------------------------------------------------------------
# 3) Anchor sizes — same as v1/v2 (checkpoint-compatible).
# ---------------------------------------------------------------------------
model.proposal_generator.anchor_generator.sizes = [
    [16],
    [32],
    [64],
    [128],
    [256],
]

# ---------------------------------------------------------------------------
# 4) Cascade matchers: allow_low_quality_matches on ALL 3 stages.
#    v2_resume only had it on stage 0. Now every GT Gun box gets a
#    positive proposal even in the stricter later stages. This is critical
#    for tiny Gun boxes that can't achieve high IoU with any proposal.
# ---------------------------------------------------------------------------
model.roi_heads.proposal_matchers = [
    L(Matcher)(thresholds=[0.4], labels=[0, 1], allow_low_quality_matches=True),
    L(Matcher)(thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=True),
    L(Matcher)(thresholds=[0.6], labels=[0, 1], allow_low_quality_matches=True),
]

# ---------------------------------------------------------------------------
# 5) Soft-NMS — same as v2_resume.
# ---------------------------------------------------------------------------
model.roi_heads.use_soft_nms = True
model.roi_heads.method = "linear"
model.roi_heads.iou_threshold = 0.2
model.roi_heads.sigma = 0.5
model.roi_heads.class_wise = True

# ---------------------------------------------------------------------------
# 6) RPN proposal capacity — same as v1/v2.
# ---------------------------------------------------------------------------
model.proposal_generator.pre_nms_topk = (4000, 2000)
model.proposal_generator.post_nms_topk = (2000, 2000)

# ---------------------------------------------------------------------------
# 7) ROI sampling: higher positive_fraction (0.75 vs 0.50).
#    With a tiny dataset and rare Gun class, cramming more positives into
#    each ROI batch gives Gun more gradient per training step.
# ---------------------------------------------------------------------------
model.roi_heads.batch_size_per_image = 512
model.roi_heads.positive_fraction = 0.75

# ---------------------------------------------------------------------------
# 8) Stronger per-class focal loss.
#
#    Alpha weights (indexed by class):
#      class 0 = Armed   → 0.15 (dominant, reduce weight)
#      class 1 = Unarmed → 0.15 (dominant, reduce weight)
#      class 2 = Gun     → 0.85 (rare/hard, maximize weight)
#      class 3 = BG      → 0.05 (de-weight easy background)
#
#    Gamma 3.0 (vs 2.0) further focuses on hard examples → Gun benefits
#    the most since it has the lowest confidence predictions.
#
#    GIoU box regression — same as v2_resume (scale-invariant).
# ---------------------------------------------------------------------------
for _bp in model.roi_heads.box_predictors:
    _bp.use_focal_loss = True
    _bp.focal_loss_alpha = [0.15, 0.15, 0.85, 0.05]
    _bp.focal_loss_gamma = 3.0
    _bp.box_reg_loss_type = "giou"

# ---------------------------------------------------------------------------
# 9) COSINE LR schedule with warm restart.
#
#    Base LR = 1e-5 (10× the v1 LR of 1e-6, and this time it WILL be used
#    because we're NOT resuming optimizer state).
#
#    Cosine annealing: LR stays near 1e-5 for the first ~60% of training
#    then gradually decays. MUCH better than multi-step which drops by 10×
#    at hard milestones and kills Gun learning.
#
#    Warmup: 1000 iters from 0.001× base → full LR (smooth ramp-up to
#    prevent instability from the higher LR).
# ---------------------------------------------------------------------------
optimizer.lr = 1e-5

train.max_iter = 50000
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1.0,
        end_value=0.01,
    ),
    warmup_length=1000 / train.max_iter,  # 1000 iters warmup
    warmup_factor=0.001,
)

# ---------------------------------------------------------------------------
# 10) Copy-paste augmentation for Gun instances.
#
#     This uses a custom DatasetMapper that pastes random Gun crops onto
#     training images. It's the most effective data-side technique for
#     boosting rare small objects.
#
#     Import and set the mapper class. The mapper will automatically load
#     Gun crops from the training annotations at initialization.
#
#     IMPORTANT: we capture the augmentations list BEFORE reassigning
#     the mapper to avoid a circular LazyCall reference.
# ---------------------------------------------------------------------------
from detectron2.data.copy_paste_mapper import CopyPasteDatasetMapper

_augmentations = dataloader.train.mapper.augmentations

dataloader.train.mapper = L(CopyPasteDatasetMapper)(
    is_train=True,
    augmentations=_augmentations,
    image_format="BGR",
    use_instance_mask=False,
    recompute_boxes=False,
    # Copy-paste specific settings
    copy_paste_prob=0.25,        # 25% chance of pasting Gun crops per image (was 0.5 — too aggressive)
    max_paste_instances=2,       # paste up to 2 Gun crops per image (was 3)
    gun_category_id=2,           # contiguous class id for Gun
    annotations_json="/kaggle/working/OriginalDataset/annotations/armed_train.json",
    image_root="/kaggle/working/OriginalDataset",
)
