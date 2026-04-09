# A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens (CVPR 2026 Highlight)

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b)](https://arxiv.org/abs/2604.04913)&nbsp;
[![Models](https://img.shields.io/badge/Hugging_Face-Models-FFD21E?labelColor=555)](https://huggingface.co/collections/Amazon-FAR/deltatok)

DeltaTok represents each video frame as a single token by compressing the frame-to-frame change in vision foundation model features. DeltaWorld uses these tokens to generate diverse plausible futures.

## Model Zoo

All models operate at 512x512 resolution with a frozen [DINOv3](https://github.com/facebookresearch/dinov3) ViT-B backbone (not included). The released DeltaTok and DeltaWorld are trained on Kinetics-700, while the paper uses a larger dataset. See [Training & Evaluation](#training--evaluation) and [Example Training Resources](#example-training-resources) for reproduction.

### Task Heads

Evaluation heads for downstream tasks. Present-time metrics (on ground-truth features):

| Model | Dataset | Metric | Download |
|-------|---------|--------|---------|
| Segmentation | VSPW | mIoU: 58.4 | [![Download](https://img.shields.io/badge/Download-seg--head--vspw-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/seg-head-vspw) |
| Segmentation | Cityscapes | mIoU: 70.5 | [![Download](https://img.shields.io/badge/Download-seg--head--cityscapes-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/seg-head-cityscapes) |
| Depth | KITTI | RMSE: 2.79 | [![Download](https://img.shields.io/badge/Download-depth--head--kitti-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/depth-head-kitti) |
| RGB | ImageNet | — | [![Download](https://img.shields.io/badge/Download-rgb--head--imagenet-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/rgb-head-imagenet) |

### DeltaTok (Tokenizer) [![Download](https://img.shields.io/badge/Download-deltatok--kinetics-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/deltatok-kinetics)

ViT-B encoder and decoder trained on Kinetics-700. Metrics show downstream task performance after encoding and decoding through DeltaTok.

| Horizon | VSPW mIoU | Cityscapes mIoU | KITTI RMSE |
|---------|-----------------|-----------|------------|
| Short (1 frame) | 58.6 | 69.6 | 2.78 |
| Mid (3 frames) | 58.5 | 67.9 | 2.86 |

### DeltaWorld (Predictor) [![Download](https://img.shields.io/badge/Download-deltaworld--kinetics-FFD21E?labelColor=555)](https://huggingface.co/Amazon-FAR/deltaworld-kinetics)

ViT-B predictor trained on Kinetics-700. Evaluation generates 20 random samples. Best selects the closest to ground truth, mean averages all predicted features before evaluation.

| Horizon | Mode | VSPW mIoU | Cityscapes mIoU | KITTI RMSE |
|---------|------|-----------------|-----------|------------|
| Short (1 frame) | *Copy last* | *51.2* | *53.5* | *3.76* |
| Short (1 frame) | Best (mean) | 56.3 (54.2) | 66.2 (64.2) | 2.95 (3.32) |
| Mid (3 frames) | *Copy last* | *44.3* | *39.6* | *4.86* |
| Mid (3 frames) | Best (mean) | 51.5 (46.6) | 55.3 (49.5) | 3.71 (4.74) |

## Setup

Requires [Miniconda](https://docs.anaconda.com/miniconda/) (or Anaconda).

```bash
conda create -n deltatok python=3.14.2
conda activate deltatok
pip install -r requirements.txt
wandb login  # for experiment logging via Weights & Biases
```

## Data Preparation

In the config files under `configs/`, uncomment the dataset sections you need and replace `/path/to/<dataset>` with your absolute paths. Training requires `train_dataset_cfg` (Kinetics), evaluation requires `val_datasets_cfg` (VSPW, Cityscapes, KITTI).

### Kinetics-700 (training, ~1.2 TB)

```bash
mkdir -p kinetics/train
wget -i https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt -P k700_tars/
for f in k700_tars/*.tar.gz; do tar -xzf "$f" -C kinetics/train; done
```

> Pre-extracted frames (as a directory of frame folders or a zip archive) are also supported for faster data loading. See [`datasets/kinetics.py`](datasets/kinetics.py) for details.

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

Training and evaluation use [Lightning CLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html). See [Notes](#notes) for GPU scaling.

### Training DeltaTok (Tokenizer)

**Stage 1: Pre-train at 256px**
```bash
python main.py fit -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --data.frame_size=256 \
  --model.compile_mode=default \
  --trainer.max_steps=1000000
```

**Stage 2: High-resolution fine-tune at 512px** (from stage 1 checkpoint, step counter resets)
```bash
python main.py fit -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --model.lr=1e-4 \
  --model.compile_mode=default \
  --trainer.max_steps=500000 \
  --model.ckpt_path=path/to/stage1/last.ckpt
```

**Stage 3-4: LR cooldowns** (resume full training state, only change LR)
```bash
# Stage 3
python main.py fit -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --model.lr=1e-5 \
  --model.compile_mode=default \
  --trainer.max_steps=550000 \
  --ckpt_path=path/to/stage2/last.ckpt

# Stage 4
python main.py fit -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --model.lr=1e-6 \
  --model.compile_mode=default \
  --trainer.max_steps=600000 \
  --ckpt_path=path/to/stage3/last.ckpt
```

### Training DeltaWorld (Predictor)

Requires a trained DeltaTok checkpoint: either the [released tokenizer](https://huggingface.co/Amazon-FAR/deltatok-kinetics) (`pytorch_model.bin`) or `last.ckpt` from your own training.

```bash
python main.py fit -c configs/deltaworld_vitb_dinov3_vitb_kinetics.yaml \
  --model.network.tokenizer.ckpt_path=path/to/deltatok-kinetics/pytorch_model.bin \
  --model.compile_mode=default \
  --trainer.max_steps=300000
```

**LR cooldown** (resume full training state, only change LR)
```bash
python main.py fit -c configs/deltaworld_vitb_dinov3_vitb_kinetics.yaml \
  --model.lr=1e-5 \
  --model.compile_mode=default \
  --trainer.max_steps=305000 \
  --ckpt_path=path/to/deltaworld/last.ckpt
```

### Evaluation

```bash
# DeltaTok
python main.py validate -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --model.ckpt_path=path/to/deltatok-kinetics/pytorch_model.bin

# DeltaWorld (requires both tokenizer and predictor checkpoints)
python main.py validate -c configs/deltaworld_vitb_dinov3_vitb_kinetics.yaml \
  --model.ckpt_path=path/to/deltaworld-kinetics/pytorch_model.bin \
  --model.network.tokenizer.ckpt_path=path/to/deltatok-kinetics/pytorch_model.bin
```

> Task head paths are configured in the config files (commented out by default). Uncomment and set paths for each task head you have. Omit any to skip that dataset's metrics. Download pre-trained task heads from the [Model Zoo](#task-heads).

### Notes

- **GPU scaling**: The default config uses 8 GPUs with batch size 128 (effective 1024). Scale `--data.batch_size` to keep the effective batch size at 1024 when changing `--trainer.devices` or `--trainer.num_nodes`. Gradient accumulation (`--trainer.accumulate_grad_batches`) can also be used to reach the target batch size with fewer GPUs.
- **`--ckpt_path` vs `--model.ckpt_path`**: `--ckpt_path` resumes full training state (model, optimizer, step) from a Lightning checkpoint. `--model.ckpt_path` loads model weights only (for evaluation or starting a new stage). The released checkpoints contain model weights only.

## Example Training Resources

Training times and memory measured on NVIDIA H200 GPUs with bf16 mixed precision and torch.compile. The GPU counts below are what we used; any configuration that maintains an effective batch size of 1024 works (see [Notes](#notes)).

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
