# EVA-02 COCO Training (BSL, 10k steps, BS=2, LR=5e-7)

This guide describes how to set up the repo and train the EVA-02 COCO detection model
using the custom config:

`projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7_10k_bs2_lr5e7.py`

## 1) Environment setup

Follow the EVA-02 "asuka" setup first:

```bash
conda create --name asuka python=3.8 -y
conda activate asuka

# If you already cloned the repo, skip the clone step
git clone git@github.com:baaivision/EVA.git
cd EVA-02/asuka

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

Install Apex and xFormers (required by EVA-02):

- Apex: [https://github.com/NVIDIA/apex#linux](https://github.com/NVIDIA/apex#linux)
- xFormers: [https://github.com/facebookresearch/xformers#installing-xformers](https://github.com/facebookresearch/xformers#installing-xformers)

## 2) Detection dependencies

From `EVA-02/det`:

```bash
pip install mmcv==1.7.1
python -m pip install -e .
```

## 3) COCO dataset

Prepare COCO in Detectron2 format and set `DETECTRON2_DATASETS` to the dataset root.

Expected structure:

```
DETECTRON2_DATASETS/
├── coco
└── ...
```

See Detectron2 dataset instructions:
[https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html)

## 4) Pretrained checkpoint

Download the EVA02 BSL COCO checkpoint:

`eva02_B_coco_bsl.pth`
[https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_B_coco_bsl.pth](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_B_coco_bsl.pth)

## 5) Train on COCO (custom config)

From `EVA-02/det`:

```bash
python tools/lazyconfig_train_net.py \
  --num-gpus 1 \
  --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7_10k_bs2_lr5e7.py \
  train.init_checkpoint=/path/to/eva02_B_coco_bsl.pth \
  train.output_dir=/path/to/output
```

## 6) Notes

- This config uses: image size 1024, batch size 2, base LR 5e-7, 10k iters.
- If you use multiple GPUs, increase `--num-gpus` and adjust
`dataloader.train.total_batch_size` in the config accordingly.

