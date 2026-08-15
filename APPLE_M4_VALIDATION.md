# DeltaTok validation on Apple Silicon M4

This guide reproduces a local DeltaTok validation run on an Apple Silicon M4
Mac using PyTorch's MPS backend. It uses a 78 MB subset of the official KITTI
Eigen-test data so the validation data stays below 1 GB.

The subset contains all 77 RGB frames and 67 annotated depth maps from
`2011_09_26_drive_0002_sync`. The repository's existing Eigen split selects 16
valid samples from these files. This is real KITTI validation data, but it is
not the full test set, so its aggregate metrics are not directly comparable to
the paper's full-dataset results.

## 1. Prepare Python 3.14.2

Pyenv builds Python from source. On macOS, install `xz` before Python so the
standard-library `_lzma` extension is available:

```bash
brew install xz

env \
  LDFLAGS="-L$(brew --prefix xz)/lib" \
  CPPFLAGS="-I$(brew --prefix xz)/include" \
  PKG_CONFIG_PATH="$(brew --prefix xz)/lib/pkgconfig" \
  pyenv install -s 3.14.2
```

Verify the base interpreter before creating the virtual environment:

```bash
~/.pyenv/versions/3.14.2/bin/python -c \
  "import lzma, ssl, sqlite3, bz2; print('Python build OK')"
```

Create the project environment and install the Apple M4 requirements:

```bash
cd /path/to/deltatok
~/.pyenv/versions/3.14.2/bin/python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-apple-m4.txt --timeout 120 --retries 8
python -m pip check
```

`requirements.txt` uses `decord2` only on Apple Silicon macOS because the
original `decord==0.6.0` does not provide a Python 3.14 macOS arm64 wheel.
`decord2` exposes the same `decord` import package.

## 2. Authenticate with Hugging Face

Accept the license for
[`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
then log in without putting the token in this repository:

```bash
.venv/bin/hf auth login
.venv/bin/hf auth whoami
```

## 3. Download model weights

Keep the Hub cache and released checkpoints inside the project directory:

```bash
mkdir -p checkpoints

HF_HUB_CACHE="$PWD/.cache/huggingface/hub" \
  .venv/bin/hf download Amazon-FAR/deltatok-kinetics pytorch_model.bin \
  --local-dir checkpoints/deltatok-kinetics

HF_HUB_CACHE="$PWD/.cache/huggingface/hub" \
  .venv/bin/hf download Amazon-FAR/depth-head-kitti pytorch_model.bin \
  --local-dir checkpoints/depth-head-kitti

HF_HUB_CACHE="$PWD/.cache/huggingface/hub" \
  .venv/bin/hf download facebook/dinov3-vitb16-pretrain-lvd1689m
```

Do not set `XDG_CACHE_HOME` while running the authenticated downloads or
validation. Doing so can redirect Hugging Face's token lookup away from the
location populated by `hf auth login`.

## 4. Prepare the sub-1-GB KITTI subset

The preparation script uses HTTP range requests to extract only one drive from
the official KITTI raw and annotated-depth ZIP archives:

```bash
.venv/bin/python scripts/prepare_kitti_m4_subset.py
```

The script is idempotent: files with the expected uncompressed size are
skipped. Verify the result:

```bash
find validation_data/kitti/2011_09_26/2011_09_26_drive_0002_sync/image_02/data \
  -type f -name '*.png' | wc -l
find validation_data/kitti/val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02 \
  -type f -name '*.png' | wc -l
du -sh validation_data/kitti
```

Expected counts are 77 RGB frames, 67 depth maps, and approximately 78 MB on
disk.

## 5. Configure local paths

Copy `.env.example` to the ignored `.env` file and set the two KITTI paths to
absolute paths:

```dotenv
KINETICS_ROOT=
VSPW_ROOT=
CITYSCAPES_ROOT=
KITTI_ROOT=/absolute/path/to/deltatok/validation_data/kitti

VSPW_HEAD_PATH=
CITYSCAPES_HEAD_PATH=
KITTI_HEAD_PATH=/absolute/path/to/deltatok/checkpoints/depth-head-kitti/pytorch_model.bin
RGB_HEAD_PATH=
```

## 6. Verify MPS

```bash
.venv/bin/python -c \
  "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```

Both values should be `True` when run directly on the Mac.

## 7. Run a one-batch smoke test

```bash
HF_HUB_CACHE="$PWD/.cache/huggingface/hub" \
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
WANDB_MODE=disabled \
.venv/bin/python main.py validate \
  -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --model.ckpt_path=checkpoints/deltatok-kinetics/pytorch_model.bin \
  --trainer.accelerator=mps \
  --trainer.devices=1 \
  --trainer.precision=32-true \
  --trainer.limit_val_batches=1 \
  --trainer.logger=false \
  --data.num_workers=0 \
  --model.num_plots=0
```

## 8. Run the complete 16-sample subset

Remove only the `--trainer.limit_val_batches=1` option:

```bash
HF_HUB_CACHE="$PWD/.cache/huggingface/hub" \
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
WANDB_MODE=disabled \
.venv/bin/python main.py validate \
  -c configs/deltatok_vitb_dinov3_vitb_kinetics.yaml \
  --model.ckpt_path=checkpoints/deltatok-kinetics/pytorch_model.bin \
  --trainer.accelerator=mps \
  --trainer.devices=1 \
  --trainer.precision=32-true \
  --trainer.logger=false \
  --data.num_workers=0 \
  --model.num_plots=0
```

The runtime overrides adapt the repository's eight-NVIDIA-GPU training
defaults to one MPS device. They do not alter the architecture, weights, input
resolution, evaluation horizons, or metric implementations.

## Reference result from an M4 run

The full subset completed in 2 minutes 25 seconds and produced:

| Metric | Value |
| --- | ---: |
| Overall feature loss | 0.0034992 |
| Short-horizon (`h1`) feature loss | 0.0030226 |
| Mid-horizon (`h3`) feature loss | 0.0039758 |
| Short-horizon depth RMSE | 1.9014 |
| Mid-horizon step 1 depth RMSE | 13.4994 |
| Mid-horizon step 2 depth RMSE | 13.4403 |
| Mid-horizon step 3 depth RMSE | 13.5156 |

The three-frame evaluation feeds reconstructed features back into subsequent
steps, so errors can accumulate. The values above describe this small single-
drive subset, not the paper's complete KITTI evaluation.

## Known warnings

- PyTorch reports that pinned memory is unsupported by MPS. This is harmless
  for this validation run.
- OpenCV and Decord may warn that both bundle FFmpeg AVFoundation classes. The
  warning did not affect the completed evaluation, so neither binary package
  was modified.
- If `torchvision` fails with `ModuleNotFoundError: _lzma`, rebuild the pyenv
  Python after installing Homebrew `xz`, then recreate `.venv`.
