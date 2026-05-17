# SPD-Conv Integration in EVA-02 SimpleFeaturePyramid

## Overview

This document describes the integration of **Space-to-Depth (SPD)** convolution into the
**SimpleFeaturePyramid (SFP)** neck of the EVA-02 detection pipeline. SPD replaces all
max-pooling-based downsampling operations in the neck, preserving fine-grained spatial
information that is critical for small-object detection.

---

## What is SPD?

SPD (Space-to-Depth) rearranges spatial pixels into the channel dimension:

```
Input:  (B, C, H, W)
Output: (B, 4C, H/2, W/2)
```

Instead of discarding 3 out of 4 pixels (as MaxPool does), SPD keeps **all** pixels by
interleaving them into channels. A subsequent 1×1 convolution compresses the channels back.

```
┌────────────┐        ┌─────────────┐        ┌────────────┐
│  (B,C,H,W) │──SPD──▸│ (B,4C,H/2,  │──1×1──▸│ (B,C,H/2,  │
│             │        │      W/2)   │  Conv   │      W/2)  │
└────────────┘        └─────────────┘        └────────────┘
```

---

## Architecture Comparison

### Before (MaxPool-based downsampling)

```
                        ViT Backbone
                    ┌──────────────────┐
  Image (3,1024,1024)│   PatchEmbed     │
                    │   12× ViT Blocks │
                    │   (embed_dim=768)│
                    └───────┬──────────┘
                            │
                   Single feature map
                    (B, 768, 64, 64)
                     stride = 16
                            │
           ┌────────────────┼────────────────┐
           │                │                │
     ┌─────▼──────┐  ┌─────▼──────┐  ┌──────▼─────┐
     │ConvTranspose│  │ConvTranspose│  │  Identity   │
     │  ×2 (4× up) │  │  ×1 (2× up) │  │  (1× same) │
     │ 768→192     │  │ 768→384     │  │  768→768    │
     └─────┬──────┘  └─────┬──────┘  └──────┬─────┘
           │                │                │
     ┌─────▼──────┐  ┌─────▼──────┐  ┌──────▼─────┐
     │ 1×1 + 3×3  │  │ 1×1 + 3×3  │  │ 1×1 + 3×3  │
     │  Conv+LN   │  │  Conv+LN   │  │  Conv+LN   │
     └─────┬──────┘  └─────┬──────┘  └──────┬─────┘
           │                │                │
        p2 (256ch)       p3 (256ch)       p4 (256ch)
        stride=4         stride=8         stride=16
                                              │
                                     ┌────────┼────────┐
                                     │                  │
                               ┌─────▼──────┐    ┌─────▼──────┐
                               │  ❌ MaxPool  │    │  ❌ MaxPool  │
                               │  (k=2,s=2)  │    │  (k=1,s=2)  │
                               └─────┬──────┘    └─────┬──────┘
                                     │                  │
                               ┌─────▼──────┐          │
                               │ 1×1 + 3×3  │          │
                               │  Conv+LN   │          │
                               └─────┬──────┘          │
                                     │                  │
                                  p5 (256ch)         p6 (256ch)
                                  stride=32          stride=64
```

**Problem:** MaxPool discards 75% of pixel values in each 2×2 window. This causes
information loss, especially harmful for small objects and fine spatial details.

---

### After (SPD-based downsampling)

```
                        ViT Backbone
                    ┌──────────────────┐
  Image (3,1024,1024)│   PatchEmbed     │
                    │   12× ViT Blocks │
                    │   (embed_dim=768)│
                    └───────┬──────────┘
                            │
                   Single feature map
                    (B, 768, 64, 64)
                     stride = 16
                            │
           ┌────────────────┼────────────────┐
           │                │                │
     ┌─────▼──────┐  ┌─────▼──────┐  ┌──────▼─────┐
     │ConvTranspose│  │ConvTranspose│  │  Identity   │
     │  ×2 (4× up) │  │  ×1 (2× up) │  │  (1× same) │
     │ 768→192     │  │ 768→384     │  │  768→768    │
     └─────┬──────┘  └─────┬──────┘  └──────┬─────┘
           │                │                │
     ┌─────▼──────┐  ┌─────▼──────┐  ┌──────▼─────┐
     │ 1×1 + 3×3  │  │ 1×1 + 3×3  │  │ 1×1 + 3×3  │
     │  Conv+LN   │  │  Conv+LN   │  │  Conv+LN   │
     └─────┬──────┘  └─────┬──────┘  └──────┬─────┘
           │                │                │
        p2 (256ch)       p3 (256ch)       p4 (256ch)
        stride=4         stride=8         stride=16
                                              │
                                     ┌────────┼────────┐
                                     │                  │
                               ┌─────▼──────┐    ┌─────▼──────┐
                               │  ✅ SPD     │    │  ✅ SPD     │
                               │ 768→3072ch  │    │ 256→1024ch  │
                               │  ↓ 1×1 Conv │    │  ↓ 1×1 Conv │
                               │ 3072→768ch  │    │ 1024→256ch  │
                               └─────┬──────┘    └─────┬──────┘
                                     │                  │
                               ┌─────▼──────┐          │
                               │ 1×1 + 3×3  │          │
                               │  Conv+LN   │          │
                               └─────┬──────┘          │
                                     │                  │
                                  p5 (256ch)         p6 (256ch)
                                  stride=32          stride=64
```

**Benefit:** SPD preserves **100%** of spatial information by rearranging pixels into
channels, then uses a learned 1×1 convolution to compress — letting the network
decide what to keep rather than hard-coding max selection.

---

## Files Modified

| File | Change |
|------|--------|
| `detectron2/modeling/backbone/vit.py` | Added `SpaceToDepth` module, `LastLevelSPD` class; replaced `MaxPool2d` with SPD in `scale==0.5` branch |
| `detectron2/modeling/backbone/__init__.py` | Exported `SpaceToDepth`, `LastLevelSPD` |
| `detectron2/modeling/__init__.py` | Added `SpaceToDepth`, `LastLevelSPD` to top-level exports |
| `configs/common/models/mask_rcnn_vitdet.py` | Switched `top_block` from `LastLevelMaxPool` to `LastLevelSPD` |

---

## Code Changes

### SpaceToDepth module (new)

```python
class SpaceToDepth(nn.Module):
    """Rearrange spatial pixels into channel dimension (2x downscale, 4x channels)."""

    def forward(self, x):
        return torch.cat(
            [x[..., ::2, ::2], x[..., 1::2, ::2],
             x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1
        )
```

### SimpleFeaturePyramid scale=0.5 branch

**Before:**
```python
elif scale == 0.5:
    layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
```

**After:**
```python
elif scale == 0.5:
    layers = [
        SpaceToDepth(),
        Conv2d(dim * 4, dim, kernel_size=1, bias=use_bias, norm=get_norm(norm, dim)),
    ]
```

### LastLevelSPD (replaces LastLevelMaxPool)

**Before:**
```python
class LastLevelMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]
```

**After:**
```python
class LastLevelSPD(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"
        self.spd = SpaceToDepth()
        self.conv = Conv2d(in_channels * 4, in_channels, kernel_size=1,
                           bias=False, norm=get_norm("LN", in_channels))

    def forward(self, x):
        return [self.conv(self.spd(x))]
```

---

## Tensor Shape Flow

| Stage | Before (MaxPool) | After (SPD) |
|-------|-------------------|-------------|
| ViT output | `(B, 768, 64, 64)` | `(B, 768, 64, 64)` |
| p2 (scale 4.0) | `(B, 256, 256, 256)` | `(B, 256, 256, 256)` |
| p3 (scale 2.0) | `(B, 256, 128, 128)` | `(B, 256, 128, 128)` |
| p4 (scale 1.0) | `(B, 256, 64, 64)` | `(B, 256, 64, 64)` |
| p5 (scale 0.5) | MaxPool → `(B, 768, 32, 32)` → Conv → `(B, 256, 32, 32)` | SPD → `(B, 3072, 32, 32)` → Conv1×1 → `(B, 768, 32, 32)` → Conv → `(B, 256, 32, 32)` |
| p6 (top_block) | max_pool2d → `(B, 256, 16, 16)` | SPD → `(B, 1024, 16, 16)` → Conv1×1 → `(B, 256, 16, 16)` |

Final output shapes for both architectures are **identical** — the downstream RPN and
Cascade ROI Heads require no changes.

---

## Why This Matters for Detection

- **Small objects** occupy very few pixels at low-resolution feature maps (p5, p6).
  MaxPool aggressively discards spatial info, making small objects harder to detect.
- **SPD** preserves all spatial information and lets the learned 1×1 conv decide what
  to keep — acting as a trainable, information-preserving downsampler.
- **No extra hyperparameters** — SPD is parameter-free; only the 1×1 conv adds learnable
  weights.
- **Minimal overhead** — the 1×1 conv adds negligible FLOPs compared to the ViT backbone.
