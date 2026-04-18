import logging
import math
import os
from collections import defaultdict

import lightning
import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import grad_norm
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.regression import MeanSquaredError

import wandb
from datasets.kitti import MAX_DEPTH as KITTI_MAX_DEPTH
from datasets.kitti import MIN_DEPTH as KITTI_MIN_DEPTH
from models.task_heads import DepthHead, RGBHead, SegHead

TASK_HEAD_KEY = {SegHead: "seg", DepthHead: "depth"}
GT_METRICS = {
    "seg": ["miou"],
    "depth": ["rmse"],
}
LABEL_VALID = {
    "seg": lambda labels: (labels != 255).any(),
    "depth": lambda labels: (labels > 0).any(),
}


def _valid_depth(
    pred_depth: torch.Tensor, gt_depth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = gt_depth > 0
    return pred_depth.squeeze(1)[valid], gt_depth[valid]


METRIC_FACTORIES = {
    "seg_miou": lambda h: MulticlassJaccardIndex(
        h.head.out_channels, ignore_index=255, validate_args=False
    ),
    "depth_rmse": lambda h: MeanSquaredError(squared=False),
}
METRIC_CALLS = {
    "seg_miou": lambda m, task_preds, labels: m(task_preds, labels),
    "depth_rmse": lambda m, task_preds, labels: m(*_valid_depth(task_preds, labels)),
}


def align_to_task_output(
    tensor: torch.Tensor,
    tgt_size: tuple[int, ...],
    overlap: bool,
    is_wide: bool | None,
    is_second_crop: bool = False,
) -> torch.Tensor:
    tgt_h, tgt_w = tgt_size[-2:]
    if not overlap:
        return F.interpolate(tensor, (tgt_h, tgt_w), mode="bilinear")
    square_size = max(tgt_h, tgt_w)
    up = F.interpolate(tensor, (square_size, square_size), mode="bilinear")
    if not is_second_crop:
        return up[..., :tgt_h, :tgt_w]
    if is_wide:
        return up[..., :tgt_h, square_size - tgt_w :]
    return up[..., square_size - tgt_h :, :tgt_w]


def upsample_to_labels(
    tensor: torch.Tensor,
    tgt_shape: tuple[int, ...],
    overlap: bool,
    is_wide: bool | None,
) -> torch.Tensor:
    half = tensor.shape[0] // 2
    first_crop = align_to_task_output(
        tensor[:half], tgt_shape, overlap, is_wide, is_second_crop=False
    )
    second_crop = align_to_task_output(
        tensor[half:], tgt_shape, overlap, is_wide, is_second_crop=True
    )
    return torch.cat([first_crop, second_crop], dim=0)


def split_into_square_crops(
    frames: torch.Tensor, labels: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None, bool, bool | None]:
    """Split rectangular frames/labels into square crops.

    Frames: two overlapping squares at corners.
    Labels: two non-overlapping halves at midpoint (no pixel evaluated twice).
    """
    h, w = frames.shape[-2:]
    if h == w:
        return frames, labels, False, None

    is_wide = w > h
    crop_size = min(h, w)

    crop1 = frames[..., :crop_size, :crop_size]
    crop2 = frames[..., w - crop_size :] if is_wide else frames[..., h - crop_size :, :]
    frames = torch.cat([crop1, crop2], dim=0)

    if labels is not None:
        label_h, label_w = labels.shape[-2:]
        label_mid = label_w // 2 if is_wide else label_h // 2
        label1 = labels[..., :label_mid] if is_wide else labels[..., :label_mid, :]
        label2 = labels[..., label_mid:] if is_wide else labels[..., label_mid:, :]

        if label1.shape != label2.shape:
            pad = [0, 1, 0, 0] if is_wide else [0, 0, 0, 1]
            label1 = F.pad(label1, pad, value=255 if labels.dtype == torch.int64 else 0)

        labels = torch.cat([label1, label2], dim=0)

    overlap = (max(w, h) / min(w, h)) < 2.0
    return frames, labels, overlap, is_wide


def preprocess_validation_batch(
    batch: tuple, frame_size: int | tuple[int, int]
) -> tuple:
    frames, timestamps, labels, sample_idx = batch[:4]
    overlap, is_wide = False, None

    if isinstance(
        frame_size, int
    ):  # int preserves aspect ratio, so split to standardize
        frames, labels, overlap, is_wide = split_into_square_crops(frames, labels)
        if is_wide is not None:
            timestamps = timestamps.repeat(2, 1)
            sample_idx = sample_idx.repeat(2)

    return frames, timestamps, labels, sample_idx, overlap, is_wide


def load_sd(module: nn.Module, sd: dict) -> torch._C._IncompatibleKeys:
    sd = sd.get("state_dict", sd)
    module = getattr(module, "_orig_mod", module)
    module_sd = module.state_dict()

    used = {}
    unmapped = []
    for ckpt_key, tensor in sd.items():
        if ckpt_key not in module_sd:
            continue
        if tensor.shape != module_sd[ckpt_key].shape:
            unmapped.append(ckpt_key)
            continue
        used[ckpt_key] = tensor

    missing = [
        n for n, p in module.named_parameters(remove_duplicate=False)
        if p.requires_grad and n not in used
    ]
    if missing or unmapped:
        raise RuntimeError(f"missing={missing}, unmapped={unmapped}")
    return module.load_state_dict(used, strict=False)


def to_numpy_img(frame: torch.Tensor) -> np.ndarray:
    return frame.permute(1, 2, 0).cpu().float().div(255.0).numpy()


def feats_to_pca(
    tgt_feats: torch.Tensor, pred_feats: torch.Tensor, h: int, w: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    def to_grid(f: torch.Tensor) -> torch.Tensor:
        return f.transpose(1, 2).reshape(-1, f.shape[2], h, w)

    return pca(to_grid(tgt_feats), to_grid(pred_feats))


def pca(
    tgt_grid: torch.Tensor, pred_grid: torch.Tensor
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    tgt_grid = torch.nan_to_num(tgt_grid).float()
    pred_grid = torch.nan_to_num(pred_grid).float()
    num_channels, h, w = tgt_grid.shape[1:]
    X = tgt_grid.permute(0, 2, 3, 1).reshape(-1, num_channels).cpu()
    if X.std() < 1e-8:
        a = np.zeros((h, w, 3), dtype=np.float32)
        return [a] * len(tgt_grid), [a] * len(pred_grid)
    with torch.autocast("cuda", enabled=False):
        mu = X.mean(0, keepdim=True)
        Xc = X - mu
        Cx = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
        Cx.diagonal().add_(1e-6)
        Wp = torch.linalg.eigh(Cx)[1][:, -3:]
        Wp[:, Wp.sum(0) < 0] *= -1
        Y_all = Xc @ Wp
        m, M = Y_all.amin(0, keepdim=True), Y_all.amax(0, keepdim=True)
        d = (M - m).clamp(min=1e-6)

    def proj(f):
        Y = ((f.cpu().flatten(1).T - mu) @ Wp).view(h, w, 3)
        return ((Y - m) / d).clamp(0, 1).numpy()

    return [proj(tgt) for tgt in tgt_grid], [proj(pred) for pred in pred_grid]


def vis_seg(
    tgt_task: torch.Tensor, gt_labels: torch.Tensor, pred_task: torch.Tensor
) -> tuple[list, list, list]:
    palette = (
        np.random.default_rng(seed=0).integers(0, 256, (256, 3), dtype=np.uint8) / 255.0
    )
    palette[255] = 0
    return (
        list(palette[gt_labels.cpu().numpy()]),
        list(palette[tgt_task.argmax(dim=1).cpu().numpy()]),
        list(palette[pred_task.argmax(dim=1).cpu().numpy()]),
    )


def vis_depth(
    tgt_task: torch.Tensor, gt_labels: torch.Tensor, pred_task: torch.Tensor
) -> tuple[list, list, list]:
    tgt_np = tgt_task.squeeze(1).cpu().numpy()
    gt_np = gt_labels.cpu().numpy()
    pred_np = pred_task.squeeze(1).cpu().numpy()
    vmax = max(tgt_np.max(), gt_np.max(), pred_np.max())
    cmap = matplotlib.colormaps["viridis"]

    def to_img(d: np.ndarray) -> np.ndarray:
        return cmap(d / vmax)[..., :3]

    return (
        [to_img(gt) if (gt > 0).any() else None for gt in gt_np],
        list(map(to_img, tgt_np)),
        list(map(to_img, pred_np)),
    )


def vis_rgb(
    tgt_task: torch.Tensor, _: torch.Tensor | None, pred_task: torch.Tensor
) -> tuple[None, list, list]:
    def to_img(t: torch.Tensor) -> list:
        return list(t.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy())

    return None, to_img(tgt_task), to_img(pred_task)


VIS_FNS = {"seg": vis_seg, "depth": vis_depth, "rgb": vis_rgb}


def prepare_frame_imgs(
    frames: torch.Tensor,
    size: tuple[int, ...],
    overlap: bool,
    is_wide: bool | None,
    is_second_crop: bool = False,
) -> list[np.ndarray]:
    aligned = align_to_task_output(
        frames.float(), size, overlap, is_wide, is_second_crop
    )
    return [to_numpy_img(frame) for frame in aligned]


def create_plot_from_rows(rows: list[list], num_cols: int) -> Image.Image:
    cell_h = cell_w = 0
    for row in rows:
        for img in row:
            if img is not None:
                cell_h = max(cell_h, img.shape[0])
                cell_w = max(cell_w, img.shape[1])
    scale = min(256 / max(cell_h, cell_w, 1), 1.0)
    cell_h, cell_w = int(cell_h * scale), int(cell_w * scale)
    num_rows = len(rows)
    grid = np.zeros((cell_h * num_rows, cell_w * num_cols, 3), dtype=np.uint8)
    for r, imgs in enumerate(rows):
        for c, img in enumerate(imgs):
            if img is not None:
                arr = (
                    (np.asarray(img, dtype=np.float32) * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                if arr.shape[0] != cell_h or arr.shape[1] != cell_w:
                    arr = np.asarray(
                        Image.fromarray(arr).resize((cell_w, cell_h), Image.BILINEAR)
                    )
                grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = arr
    return Image.fromarray(grid)


class LogCoshLoss(nn.Module):
    def forward(
        self, pred_feats: torch.Tensor, tgt_feats: torch.Tensor
    ) -> torch.Tensor:
        diff = (pred_feats - tgt_feats).abs()
        return diff + F.softplus(-2.0 * diff) - math.log(2.0)


class Base(lightning.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        ckpt_path: str | None,
        loss_fn: str,
        loss_beta: float | None = None,
        **kw,
    ):
        super().__init__()
        for key, value in kw.items():
            setattr(self, key, value)
        self._plot_imgs = []
        assert loss_fn != "smooth_l1" or loss_beta is not None, "loss_beta required for smooth_l1"
        self.criterion = {
            "mse": nn.MSELoss(reduction="none"),
            "log_cosh": LogCoshLoss(),
            "smooth_l1": nn.SmoothL1Loss(reduction="none", beta=loss_beta),
        }[loss_fn]
        self.network = network
        if ckpt_path:
            load_sd(self.network, torch.load(ckpt_path))

        head_configs = {
            "vspw": (SegHead, {"num_classes": 124}),
            "cityscapes": (SegHead, {"num_classes": 19}),
            "kitti": (
                DepthHead,
                {"min_depth": KITTI_MIN_DEPTH, "max_depth": KITTI_MAX_DEPTH},
            ),
            "rgb": (
                RGBHead,
                {
                    "norm_weight": self.network.backbone.backbone.norm.weight,
                    "norm_bias": self.network.backbone.backbone.norm.bias,
                    "img_mean": self.network.backbone.processor.image_mean,
                    "img_std": self.network.backbone.processor.image_std,
                },
            ),
        }
        self.task_heads = nn.ModuleDict()
        for name, (head_cls, head_kw) in head_configs.items():
            env_var = f"{name.upper()}_HEAD_PATH"
            path = os.environ.get(env_var)
            if not path:
                logging.info("Skipping %s head: %s not set", name, env_var)
                continue
            head = head_cls(self.network.backbone.hidden_size, **head_kw)
            load_sd(head, torch.load(path))
            head.requires_grad_(False).eval()
            self.task_heads[name] = head

        self.metrics = nn.ModuleDict()
        self.val_loss_sum, self.val_loss_count = defaultdict(float), defaultdict(int)

    def setup(self, stage: str) -> None:
        compile_on = stage == "fit" if self.use_compile is None else self.use_compile
        if compile_on:
            self.network = torch.compile(self.network)

    def configure_optimizers(self) -> dict:
        decay_types = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        )

        decay = {}
        for n, m in self.network.named_modules():
            if isinstance(m, decay_types):
                w = getattr(m, "weight", None)
                if isinstance(w, nn.Parameter) and w.requires_grad:
                    decay[f"{n}.weight" if n else "weight"] = w

        no_decay = {
            n: p
            for n, p in self.network.named_parameters()
            if p.requires_grad and n not in decay
        }

        decay_list = [decay[k] for k in sorted(decay)]
        no_decay_list = [no_decay[k] for k in sorted(no_decay)]

        opt = AdamW(
            [
                {"params": decay_list, "weight_decay": self.weight_decay},
                {"params": no_decay_list, "weight_decay": 0.0},
            ],
            self.lr,
        )

        def lr_lambda(_: int) -> float:
            step = self.trainer.global_step
            if self.lr_warmup_steps > 0 and step < self.lr_warmup_steps:
                return step / self.lr_warmup_steps
            return 1.0

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": LambdaLR(opt, lr_lambda), "interval": "step"},
        }

    def load_state_dict(
        self, state_dict: dict, strict: bool = False
    ) -> torch._C._IncompatibleKeys:
        return load_sd(self.network, state_dict)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["lr_schedulers"] = []  # reset to respect config changes on resume

    def on_train_start(self) -> None:
        # Reapply config: optimizer.load_state_dict restores checkpoint param_groups,
        # and global_step is restored after configure_optimizers
        for cfg in self.trainer.lr_scheduler_configs:
            cfg.scheduler.base_lrs = [self.lr] * len(cfg.scheduler.base_lrs)
            opt = cfg.scheduler.optimizer
            opt.param_groups[0]["weight_decay"] = self.weight_decay
            opt.param_groups[1]["weight_decay"] = 0.0
            for lmbda, base_lr, group in zip(
                cfg.scheduler.lr_lambdas,
                cfg.scheduler.base_lrs,
                opt.param_groups,
            ):
                group["lr"] = base_lr * lmbda(0)

    def on_before_optimizer_step(self, optimizer: AdamW) -> None:
        self._log_scalar(
            "total",
            grad_norm(self.network, norm_type=2)["grad_2.0_norm_total"].to(self.device),
            "grad_norm",
        )
        for name, child in getattr(
            self.network, "_orig_mod", self.network
        ).named_children():
            grads = [p.grad.flatten() for p in child.parameters() if p.grad is not None]
            if grads:
                self._log_scalar(name, torch.cat(grads).norm(2), "grad_norm")

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        pred_feats, tgt_feats = self.network(*batch, self.criterion)
        with torch.autocast(device_type="cuda", enabled=False):
            loss = self.criterion(pred_feats.float(), tgt_feats.detach().float()).mean()
        self._log_scalar("train", loss, "losses", prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.trainer.logger:
            imgs = self._plot_imgs
            if dist.is_available() and dist.is_initialized():
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, imgs)
                imgs = sum(gathered, []) if not dist.get_rank() else []
            if imgs:
                self.trainer.logger.experiment.log(
                    {key: [wandb.Image(img, file_type="jpg")] for key, img in imgs},
                )
        self._plot_imgs = []
        if self.val_loss_sum:
            avgs = [
                self.val_loss_sum[dataset] / self.val_loss_count[dataset]
                for dataset in self.val_loss_sum
            ]
            self._log_scalar("val", sum(avgs) / len(avgs), "losses", prog_bar=True)
            self.val_loss_sum.clear()
            self.val_loss_count.clear()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        network = getattr(self.network, "_orig_mod", self.network)
        checkpoint["state_dict"] = {
            k: v
            for k, v in network.state_dict().items()
            if any(
                p.requires_grad
                for p in network.get_submodule(k.rpartition(".")[0]).parameters()
            )
        }

    def _apply_head(
        self,
        feats: torch.Tensor,
        head: nn.Module,
        frame_shape: tuple[int, ...],
    ) -> torch.Tensor:
        batch_size, num_steps = feats.shape[:2]
        patch_size = self.network.backbone.patch_size
        h, w = frame_shape[-2] // patch_size, frame_shape[-1] // patch_size
        feat_map = (
            feats.reshape(-1, *feats.shape[2:])
            .transpose(1, 2)
            .reshape(-1, feats.shape[-1], h, w)
        )
        out = head(feat_map).float()
        return out.reshape(batch_size, num_steps, *out.shape[1:])

    def _log_scalar(
        self, name: str, value: torch.Tensor | float, prefix: str, **kw
    ) -> None:
        self.log(
            f"{prefix}/{name}",
            value,
            sync_dist=True,
            add_dataloader_idx=False,
            **kw,
        )

    def _log_plot_img(self, key: str, img: Image.Image) -> None:
        self._plot_imgs.append((key, img))

    def _eval_metric(
        self,
        task_preds: torch.Tensor,
        labels: torch.Tensor,
        head: nn.Module,
        dataset: str,
        prefix: str,
        key: str,
        suffix: str,
    ) -> None:
        task_key = TASK_HEAD_KEY[type(head)]
        if dataset not in self.metrics:
            self.metrics[dataset] = nn.ModuleDict()
        dataset_metrics = self.metrics[dataset]
        for metric_name in GT_METRICS[task_key]:
            metric_key = f"{task_key}_{metric_name}"
            full_key = f"{key}_{task_key}_{metric_name}{suffix}"
            if full_key not in dataset_metrics:
                dataset_metrics[full_key] = METRIC_FACTORIES[metric_key](head).to(
                    self.device
                )
            metric = dataset_metrics[full_key]
            METRIC_CALLS[metric_key](metric, task_preds, labels)
            self.log(
                f"metrics/{prefix}_{full_key}",
                metric,
                add_dataloader_idx=False,
                metric_attribute=f"metrics.{dataset}.{full_key}",
            )

    def _eval_horizon(
        self,
        pred_feats: torch.Tensor,
        tgt_feats: torch.Tensor,
        labels: torch.Tensor | None,
        ctx_len: int,
        horizon: int,
        task_head: nn.Module | None,
        frame_shape: tuple[int, ...],
        overlap: bool,
        is_wide: bool | None,
        dataset_name: str,
        log_prefix: str,
        key_suffix: str = "",
    ) -> torch.Tensor:
        losses = []
        key_base = f"c{ctx_len}_h{horizon}"
        horizon_labels = labels[:, -horizon:] if labels is not None else None
        num_preds = pred_feats.shape[1]
        task_key = TASK_HEAD_KEY[type(task_head)] if task_head is not None else None

        for t in range(num_preds):
            horizon_t = horizon - num_preds + t
            loss = self.criterion(pred_feats[:, t], tgt_feats[:, horizon_t]).mean()
            self._log_scalar(
                f"{log_prefix}_{key_base}_t{horizon_t}{key_suffix}", loss, "losses"
            )
            losses.append(loss)

            if task_key is not None and LABEL_VALID[task_key](
                horizon_labels[:, horizon_t]
            ):
                labels_t = horizon_labels[:, horizon_t]
                preds_t = upsample_to_labels(
                    self._apply_head(pred_feats[:, t : t + 1], task_head, frame_shape)[
                        :, 0
                    ],
                    labels_t.shape[-2:],
                    overlap,
                    is_wide,
                )
                self._eval_metric(
                    preds_t,
                    labels_t,
                    task_head,
                    dataset_name,
                    log_prefix,
                    f"{key_base}_t{horizon_t}",
                    key_suffix,
                )

        mean_loss = sum(losses) / len(losses)
        self._log_scalar(f"{log_prefix}_{key_base}{key_suffix}", mean_loss, "losses")
        return mean_loss

    def _eval_val_loss(
        self, log_prefix: str, dataset_name: str, losses: list[torch.Tensor]
    ) -> torch.Tensor:
        val_loss = sum(losses) / len(losses)
        self._log_scalar(log_prefix, val_loss, "losses", prog_bar=True)
        self.val_loss_sum[dataset_name] += val_loss.item()
        self.val_loss_count[dataset_name] += 1
        return val_loss

    def _plot_selected_samples(
        self, sample_idx: torch.Tensor, num_plots: int, **data
    ) -> None:
        seen: set[int] = set()
        for i, idx in enumerate(sample_idx.tolist()):
            if idx >= num_plots or idx in seen:
                continue
            seen.add(idx)
            sliced = {
                k: v[i : i + 1] if torch.is_tensor(v) else v
                for k, v in data.items()
            }
            sliced["prefix"] = f"{sliced['prefix']}_s{idx}"
            self._plot_sample(sliced)

    def _plot_sample(self, sample: dict) -> None:
        self._plot_feats(sample)
        if task_head := getattr(self.task_heads, sample["dataset_name"], None):
            self._plot_task(sample, task_head)
        if "rgb" in self.task_heads:
            self._plot_rgb(sample)

    def _plot_feats(self, sample: dict) -> None: ...

    def _plot_task(self, sample: dict, task_head: nn.Module) -> None: ...

    def _plot_rgb(self, sample: dict) -> None: ...
