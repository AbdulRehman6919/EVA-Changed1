# EVA-02 Detection Pipeline — Accuracy Improvements

All changes applied to the EVA-02 Cascade R-CNN detection pipeline for the ARMED
dataset. Each section shows the **before** and **after** state side by side.

---

## 1. Data Augmentation — Color/Photometric Transforms

**File:** `projects/ViTDet/configs/common/coco_loader_lsj_1024.py`

**Why:** The original pipeline only used geometric augmentations (flip, scale, crop).
Adding photometric jitter (contrast, brightness, saturation) forces the model to learn
color-invariant features and reduces overfitting on small custom datasets.

### Before

```python
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
```

### After

```python
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.RandomContrast)(intensity_min=0.6, intensity_max=1.4),
    L(T.RandomBrightness)(intensity_min=0.6, intensity_max=1.4),
    L(T.RandomSaturation)(intensity_min=0.5, intensity_max=1.5),
    L(T.ResizeScale)(
        min_scale=0.3, max_scale=2.5, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
```

| Parameter | Before | After |
|-----------|--------|-------|
| RandomContrast | None | 0.6 – 1.4 |
| RandomBrightness | None | 0.6 – 1.4 |
| RandomSaturation | None | 0.5 – 1.5 |
| ResizeScale min | 0.1 | 0.3 |
| ResizeScale max | 2.0 | 2.5 |

**Expected impact:** +1–3 mAP (strongest single improvement for small datasets).

---

## 2. Learning Rate Tuning

**File:** `projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7_10k_bs2_lr5e7.py`

**Why:** The original LR (5e-7) was very conservative. With SPD adding new randomly-
initialized Conv layers in the neck, a slightly higher LR helps those layers converge
faster while the backbone (with layer-wise LR decay = 0.7) stays stable.

### Before

```python
# Optimizer: AdamW, Base LR: 5e-7
optimizer.lr = 5e-7
```

### After

```python
# Optimizer: AdamW, Base LR: 1e-6 (tuned up from 5e-7 for SPD conv layers)
optimizer.lr = 1e-6
```

| Parameter | Before | After |
|-----------|--------|-------|
| Base LR | 5e-7 | 1e-6 (2× increase) |

**Expected impact:** +0.5–2 mAP.

---

## 3. Cascade IoU Thresholds

**File:** `projects/ViTDet/configs/eva2_mim_to_coco/cascade_mask_rcnn_vitdet_b_100ep.py`

**Why:** The standard thresholds [0.5, 0.6, 0.7] were designed for COCO (300K+ annotations).
For a smaller custom dataset with fewer positives and potentially noisier bounding boxes,
lower thresholds [0.4, 0.5, 0.6] give each cascade stage more positive training samples.

### Before

```python
proposal_matchers=[
    L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
    for th in [0.5, 0.6, 0.7]
],
```

### After

```python
proposal_matchers=[
    L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
    for th in [0.4, 0.5, 0.6]
],
```

| Cascade Stage | Before | After |
|---------------|--------|-------|
| Stage 1 | IoU ≥ 0.5 | IoU ≥ 0.4 |
| Stage 2 | IoU ≥ 0.6 | IoU ≥ 0.5 |
| Stage 3 | IoU ≥ 0.7 | IoU ≥ 0.6 |

**Expected impact:** +0.3–1 mAP.

---

## 4. Soft-NMS

**Files:**
- `detectron2/modeling/roi_heads/fast_rcnn.py` (implementation)
- `projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7_10k_bs2_lr5e7.py` (config)

**Why:** Hard NMS removes all overlapping detections above an IoU threshold, which
hurts recall in crowded scenes (e.g., multiple people close together). Soft-NMS
gradually decays scores of overlapping boxes instead of removing them.

### Before

`FastRCNNOutputLayers.inference()` called `fast_rcnn_inference()` with hard NMS only:

```python
return fast_rcnn_inference(
    boxes, scores, image_shapes,
    self.test_score_thresh,
    self.test_nms_thresh,
    self.test_topk_per_image,
)
```

No `use_soft_nms` parameter in `__init__`.

### After

`FastRCNNOutputLayers.__init__` now accepts Soft-NMS parameters:

```python
def __init__(self, ...,
    use_soft_nms: bool = False,
    soft_nms_method: str = "linear",
    soft_nms_sigma: float = 0.5,
    soft_nms_iou_threshold: float = 0.3,
):
```

`inference()` passes them through:

```python
return fast_rcnn_inference(
    boxes, scores, image_shapes,
    self.test_score_thresh,
    self.test_nms_thresh,
    self.test_topk_per_image,
    use_soft_nms=self.use_soft_nms,
    method=self.soft_nms_method,
    sigma=self.soft_nms_sigma,
    iou_threshold=self.soft_nms_iou_threshold,
)
```

Config enables it for all cascade stages:

```python
for _bp in model.roi_heads.box_predictors:
    _bp.use_soft_nms = True
    _bp.soft_nms_method = "linear"
    _bp.soft_nms_sigma = 0.5
    _bp.soft_nms_iou_threshold = 0.3
```

| Parameter | Before | After |
|-----------|--------|-------|
| NMS type | Hard NMS | Soft-NMS (linear) |
| IoU threshold | 0.5 (hard cutoff) | 0.3 (soft decay start) |
| Sigma | N/A | 0.5 |

**Expected impact:** +0.3–1 mAP (especially for overlapping person/gun detections).

---

## 5. Model EMA (Exponential Moving Average)

**File:** `projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7_10k_bs2_lr5e7.py`

**Why:** EMA maintains a smoothed copy of all model parameters (averaged over training
steps). This acts as implicit regularization, producing more stable and generalizable
weights. The training script already supports EMA — it just needs enabling.

### Before

```python
# (nothing — default from configs/common/train.py)
# train.model_ema.enabled = False
```

### After

```python
train.model_ema.enabled = True
train.model_ema.decay = 0.9999
train.model_ema.use_ema_weights_for_eval_only = True
```

| Parameter | Before | After |
|-----------|--------|-------|
| EMA enabled | False | True |
| Decay rate | N/A | 0.9999 |
| Use EMA for eval | N/A | True |

**Expected impact:** +0.3–1 mAP.

---

## Summary of All Changes

| # | Change | File(s) | Type | Expected mAP Gain |
|---|--------|---------|------|-------------------|
| 1 | Color augmentation (contrast, brightness, saturation) | `coco_loader_lsj_1024.py` | Data | +1–3 |
| 2 | Multi-scale range 0.3–2.5 (from 0.1–2.0) | `coco_loader_lsj_1024.py` | Data | +0.5–1 |
| 3 | LR increase 5e-7 → 1e-6 | `eva2_...bs2_lr5e7.py` | Optimizer | +0.5–2 |
| 4 | Cascade IoU [0.5,0.6,0.7] → [0.4,0.5,0.6] | `cascade_mask_rcnn_vitdet_b_100ep.py` | Head | +0.3–1 |
| 5 | Soft-NMS (linear, σ=0.5) | `fast_rcnn.py` + config | Inference | +0.3–1 |
| 6 | Model EMA (decay=0.9999) | `eva2_...bs2_lr5e7.py` | Training | +0.3–1 |

**Combined with SPD neck integration (see `SPD_INTEGRATION.md`), total expected
improvement: +2–6 mAP over baseline.**

---

## Full Pipeline After All Changes

```
Input Image (3, 1024, 1024)
    │
    ├── Augmentation Pipeline (ENHANCED)
    │   ├── RandomFlip (horizontal)
    │   ├── RandomContrast (0.6–1.4)          ← NEW
    │   ├── RandomBrightness (0.6–1.4)        ← NEW
    │   ├── RandomSaturation (0.5–1.5)        ← NEW
    │   ├── ResizeScale (0.3–2.5)             ← WIDENED
    │   └── FixedSizeCrop (1024×1024)
    │
    ▼
  ViT-B Backbone (unchanged)
    │  12 blocks, embed_dim=768, patch_size=16
    │  Window attention + Global attention
    │
    ▼
  SimpleFeaturePyramid + SPD (see SPD_INTEGRATION.md)
    │  p2 (stride 4)  ← ConvTranspose ×2
    │  p3 (stride 8)  ← ConvTranspose ×1
    │  p4 (stride 16) ← Identity
    │  p5 (stride 32) ← SPD + Conv1×1       ← CHANGED (was MaxPool)
    │  p6 (stride 64) ← SPD + Conv1×1       ← CHANGED (was MaxPool)
    │
    ▼
  RPN (2-conv head)
    │
    ▼
  Cascade ROI Heads (3 stages)
    │  Stage 1: IoU ≥ 0.4                    ← LOWERED (was 0.5)
    │  Stage 2: IoU ≥ 0.5                    ← LOWERED (was 0.6)
    │  Stage 3: IoU ≥ 0.6                    ← LOWERED (was 0.7)
    │
    ▼
  Post-Processing
    │  Soft-NMS (linear, σ=0.5)              ← CHANGED (was hard NMS)
    │
    ▼
  Output Detections (3 classes: ARMED dataset)

  Training Extras:
    ├── LR: 1e-6                             ← INCREASED (was 5e-7)
    └── Model EMA (decay=0.9999)             ← NEW
```
