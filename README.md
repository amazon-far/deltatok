# A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens

[![CVPR 2026 Highlight](https://img.shields.io/badge/CVPR_2026-Highlight-1f7a8c)](https://cvpr.thecvf.com/virtual/2026/poster/38021)&nbsp;
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b)](https://arxiv.org/abs/2604.04913)&nbsp;
[![Models](https://img.shields.io/badge/Hugging_Face-Models-FFD21E?labelColor=555)](https://huggingface.co/collections/Amazon-FAR/deltatok)

DeltaTok compresses the frame-to-frame change in vision foundation model features into a single delta token, enabling DeltaWorld to efficiently generate diverse plausible futures.

## Model Zoo

All models operate at 512x512 resolution with a frozen [DINOv3](https://github.com/facebookresearch/dinov3) ViT-B backbone. The released DeltaTok and DeltaWorld are trained on Kinetics-700, while the paper uses a larger dataset. See [Training & Evaluation](#training--evaluation) and [Example Training Resources](#example-training-resources) for reproduction.

### Task Heads

Evaluation heads for downstream tasks:

| Task | Dataset | Metric | Download |
|------|---------|--------|---------|
| Segmentation | VSPW | mIoU: 58.4 | [![Download](https://img.shields.io/badge/Download-seg--head--vspw-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/seg-head-vspw) |
| Segmentation | Cityscapes | mIoU: 70.5 | [![Download](https://img.shields.io/badge/Download-seg--head--cityscapes-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/seg-head-cityscapes) |
| Depth | KITTI | RMSE: 2.79 | [![Download](https://img.shields.io/badge/Download-depth--head--kitti-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/depth-head-kitti) |
| RGB | ImageNet | visualization only | [![Download](https://img.shields.io/badge/Download-rgb--head--imagenet-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/rgb-head-imagenet) |

### DeltaTok (Tokenizer) [![Download](https://img.shields.io/badge/Download-deltatok--kinetics-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/deltatok-kinetics)

ViT-B encoder and decoder trained on Kinetics-700. Reconstruction quality is measured by applying downstream task heads to the reconstructed features.

| Horizon | VSPW mIoU (↑) | Cityscapes mIoU (↑) | KITTI RMSE (↓) |
|---------|---------------|---------------------|----------------|
| Short (1 frame) | 58.6 | 69.6 | 2.78 |
| Mid (3 frames)* | 58.5 | 67.9 | 2.86 |

*Parallel encoding from ground-truth frames with autoregressive decoding from previous reconstructions.

### DeltaWorld (Predictor) [![Download](https://img.shields.io/badge/Download-deltaworld--kinetics-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/deltaworld-kinetics)

ViT-B predictor trained on Kinetics-700. Prediction quality is measured by applying downstream task heads to the predicted features. Cells report *best*-of-20 with *mean* in parentheses. *best* selects the sample with lowest DINOv3-feature loss to ground truth; *mean* averages DINOv3 features across all samples before evaluation.

| Method | Horizon | VSPW mIoU (↑) | Cityscapes mIoU (↑) | KITTI RMSE (↓) |
|--------|---------|---------------|---------------------|----------------|
| *Copy last (lower bound)* | *Short (1 frame)* | *51.2* | *53.5* | *3.76* |
| DeltaWorld | Short (1 frame) | 56.3 (54.2) | 66.2 (64.2) | 2.95 (3.32) |
| *Copy last (lower bound)* | *Mid (3 frames)* | *44.3* | *39.6* | *4.86* |
| DeltaWorld | Mid (3 frames) | 51.5 (46.6) | 55.3 (49.5) | 3.71 (4.74) |

## Setup

Requires [Miniconda](https://docs.anaconda.com/miniconda/) (or Anaconda), a [Weights & Biases](https://wandb.ai/) account for logging, and a [Hugging Face](https://huggingface.co/) account. Accept the license at [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) so the gated DINOv3 ViT-B backbone downloads automatically on first run.

```bash
conda create -n deltatok python=3.14.2
conda activate deltatok
pip install -r requirements.txt
wandb login
hf auth login
cp .env.example .env
```

## Data Preparation

Prepare Kinetics-700 to train from scratch, and any of VSPW, Cityscapes, or KITTI for evaluation metrics and visualizations on that dataset. For each dataset you prepare, set the corresponding `*_ROOT` path in `.env` to the absolute path of the downloaded dataset directory.

### Kinetics-700 (training, ~1.2 TB)

```bash
mkdir -p kinetics/train
wget -i https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt -P k700_tars/
for f in k700_tars/*.tar.gz; do tar -xzf "$f" -C kinetics/train; done
```

> Pre-extracted frames (as a directory of frame folders or zip archives) are also supported for faster data loading. See [`datasets/kinetics.py`](datasets/kinetics.py) for details.

### VSPW (evaluation, ~43 GB)

```bash
pip install gdown
gdown "https://drive.google.com/file/d/14yHWsGneoa1pVdULFk7cah3t-THl7yEz/view?usp=sharing" --fuzzy
tar -xf VSPW_dataset.tar  # extracts to VSPW/
```

> If `gdown` fails due to rate limiting, download `VSPW_dataset.tar` manually from the [Google Drive link](https://drive.google.com/file/d/14yHWsGneoa1pVdULFk7cah3t-THl7yEz/view?usp=sharing).

### Cityscapes (evaluation, ~325 GB)

Requires registration at the [Cityscapes website](https://www.cityscapes-dataset.com/). Set `CITYSCAPES_USERNAME` and `CITYSCAPES_PASSWORD` environment variables for headless servers, or `csDownload` will prompt interactively.

```bash
pip install cityscapesscripts
mkdir -p cityscapes
csDownload -d cityscapes gtFine_trainvaltest.zip leftImg8bit_sequence_trainvaltest.zip
cd cityscapes && unzip -q gtFine_trainvaltest.zip && unzip -q leftImg8bit_sequence_trainvaltest.zip && cd ..
```

### KITTI (evaluation, ~44 GB)

```bash
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
unzip data_depth_annotated.zip -d kitti && rm data_depth_annotated.zip
for drive in 2011_09_26_drive_{0002,0009,0013,0020,0023,0027,0029,0036,0046,0048,0052,0056,0059,0064,0084,0086,0093,0096,0101,0106,0117} 2011_09_28_drive_0002 2011_09_29_drive_0071 2011_09_30_drive_{0016,0018,0027} 2011_10_03_drive_{0027,0047}; do
  wget -P kitti "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${drive}/${drive}_sync.zip"
  unzip -o -d kitti "kitti/${drive}_sync.zip" && rm "kitti/${drive}_sync.zip"
done
```

## Training & Evaluation

Training and evaluation use [Lightning CLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html). To get evaluation metrics and visualizations on a dataset, download the pre-trained [task head](#task-heads) for that dataset and set the corresponding `*_HEAD_PATH` in `.env` to the absolute path of the downloaded file.

The effective batch size should be 1024 for both DeltaTok and DeltaWorld. It's the product of four parameters:

```
--data.batch_size × --trainer.devices × --trainer.num_nodes × --trainer.accumulate_grad_batches
```

The default config reaches this on a single node with 8 GPUs at per-GPU batch size 128 and no gradient accumulation; adjust any of the four parameters to fit your hardware. See [Example Training Resources](#example-training-resources) for the configurations we used for each stage.

### Training DeltaTok (Tokenizer)

**Stage 1: Pre-train at 256px**
```bash
python main.py fit -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --data.frame_size=256 \
  --trainer.max_steps=1000000
```

**Stage 2: High-resolution fine-tune at 512px**

`--model.ckpt_path` loads model weights only; optimizer state and step counter reset.
```bash
python main.py fit -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --model.lr=1e-4 \
  --trainer.max_steps=500000 \
  --model.ckpt_path=path/to/stage1/last.ckpt
```

**Stage 3-4: LR cooldowns**

`--ckpt_path` resumes full training state (model weights, optimizer state, step counter).
```bash
# Stage 3
python main.py fit -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --model.lr=1e-5 \
  --trainer.max_steps=550000 \
  --ckpt_path=path/to/stage2/last.ckpt

# Stage 4
python main.py fit -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --model.lr=1e-6 \
  --trainer.max_steps=600000 \
  --ckpt_path=path/to/stage3/last.ckpt
```

### Training DeltaWorld (Predictor)

Requires a DeltaTok checkpoint: either the [released one](https://huggingface.co/Amazon-FAR/deltatok-kinetics) (`pytorch_model.bin`) or one from your own training (`last.ckpt`).

```bash
python main.py fit -c configs/deltaworld_vitb_dinov3_vitb_kinetics.yaml \
  --model.network.tokenizer.ckpt_path=path/to/deltatok-kinetics/pytorch_model.bin \
  --trainer.max_steps=300000
```

**LR cooldown**
```bash
python main.py fit -c configs/deltaworld_vitb_dinov3_vitb_kinetics.yaml \
  --model.lr=1e-5 \
  --trainer.max_steps=305000 \
  --ckpt_path=path/to/deltaworld/last.ckpt
```

### Evaluation

**DeltaTok**
```bash
python main.py validate -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --model.ckpt_path=path/to/deltatok-kinetics/pytorch_model.bin
```

**DeltaWorld**

Requires both DeltaTok and DeltaWorld checkpoints.
```bash
python main.py validate -c configs/deltaworld_vitb_dinov3_vitb_kinetics.yaml \
  --model.ckpt_path=path/to/deltaworld-kinetics/pytorch_model.bin \
  --model.network.tokenizer.ckpt_path=path/to/deltatok-kinetics/pytorch_model.bin
```

## Example Training Resources

Training times and memory are measured on NVIDIA H200 GPUs. The configurations below are examples; any setup that reaches the target [effective batch size](#training--evaluation) works.

### DeltaTok

| Stage | Resolution | LR | Steps | GPUs | Batch/GPU | GPU Memory | Time |
|-------|-----------|-----|-------|------|-----------|------------|------|
| 1. Pre-train | 256 | 1e-3 | 1M | 8 | 128 | 65 GB | 82h |
| 2. Hi-res fine-tune | 512 | 1e-4 | 500k | 16 | 64 | 109 GB | 89h |
| 3. LR cooldown | 512 | 1e-5 | 50k | 16 | 64 | 109 GB | 9h |
| 4. LR cooldown | 512 | 1e-6 | 50k | 16 | 64 | 109 GB | 9h |

### DeltaWorld

| Stage | Resolution | LR | Steps | GPUs | Batch/GPU | GPU Memory | Time |
|-------|-----------|-----|-------|------|-----------|------------|------|
| 1. Train | 512 | 1e-4 | 300k | 32 | 32 | 58 GB | 32h |
| 2. LR cooldown | 512 | 1e-5 | 5k | 32 | 32 | 58 GB | <1h |

## Citation

```bibtex
@inproceedings{kerssies2026deltatok,
  title     = {A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens},
  author    = {Kerssies, Tommie and Berton, Gabriele and He, Ju and Yu, Qihang and Ma, Wufei and de Geus, Daan and Dubbelman, Gijs and Chen, Liang-Chieh},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgements

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [RAE](https://github.com/bytetriper/RAE)
- [Kinetics-700](https://github.com/cvdfoundation/kinetics-dataset)
- [VSPW](https://www.vspwdataset.com/)
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [KITTI](https://www.cvlibs.net/datasets/kitti/)
- [ImageNet](https://www.image-net.org/)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
