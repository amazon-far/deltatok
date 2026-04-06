from functools import partial

import numpy as np
import torch

from training.base import (
    TASK_HEAD_KEY,
    VIS_FNS,
    Base,
    create_plot_from_rows,
    feats_to_pca,
    prepare_frame_imgs,
    preprocess_validation_batch,
    to_numpy_img,
)


class World(Base):
    def __init__(
        self,
        network: torch.nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 4e-1,
        loss_fn: str = "smooth_l1",
        loss_beta: float = 1e-1,
        eval_ctx_lens: tuple[int, ...] = (4,),
        eval_horizons: tuple[int, ...] = (1, 3),
        eval_copy_last: bool = False,
        lr_warmup_steps: int = 5000,
        num_plots: int = 4,
        vspw_head_path: str | None = None,
        cityscapes_head_path: str | None = None,
        kitti_head_path: str | None = None,
        rgb_head_path: str | None = None,
        ckpt_path: str | None = None,
        compile_mode: str | None = None,
    ):
        super().__init__(
            **{k: v for k, v in locals().items() if k not in ("self", "__class__")}
        )
        self._init_task_heads_and_metrics(
            vspw_head_path,
            cityscapes_head_path,
            kitti_head_path,
            rgb_head_path,
        )

    def validation_step(
        self, batch: tuple, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        frames, timestamps, labels, sample_idx, overlap, is_wide = (
            preprocess_validation_batch(batch, self.trainer.datamodule.frame_size)
        )
        batch_size = frames.shape[0]

        dataset_name = self.trainer.val_dataloaders[dataloader_idx].dataset_name
        log_prefix = f"val_{dataset_name}"

        task_head = getattr(self.task_heads, dataset_name, None)

        num_frames = frames.shape[1]
        losses = []
        for ctx_len in self.eval_ctx_lens:
            for horizon in self.eval_horizons:
                if ctx_len + horizon > num_frames:
                    continue
                ctx_start = num_frames - ctx_len - horizon
                horizon_frames = frames[:, ctx_start:]
                horizon_timestamps = timestamps[:, ctx_start:]
                if self.eval_copy_last:
                    feats = self.network.backbone(horizon_frames)
                    ctx_feats, tgt_feats = (
                        feats[:, :ctx_len],
                        feats[:, ctx_len:],
                    )
                    all_rollouts = ctx_feats[:, -1, None, None].expand(
                        batch_size, 1, horizon, -1, -1
                    )
                else:
                    all_rollouts, tgt_feats, ctx_feats = self.network(
                        horizon_frames, horizon_timestamps, ctx_len=ctx_len
                    )

                mean_rollout = all_rollouts.mean(dim=1)
                eval_kw = dict(
                    tgt_feats=tgt_feats,
                    labels=labels,
                    horizon=horizon,
                    ctx_len=ctx_len,
                    frame_shape=horizon_frames.shape,
                    log_prefix=log_prefix,
                    dataset_name=dataset_name,
                    task_head=task_head,
                    overlap=overlap,
                    is_wide=is_wide,
                )
                losses.append(
                    self._eval_horizon(mean_rollout, **eval_kw, key_suffix="_mean")
                )

                if all_rollouts.shape[1] == 1:
                    best_idx = None
                    best_rollout = None
                else:
                    last_rollout = all_rollouts[:, :, -1]
                    loss = self.criterion(
                        last_rollout,
                        tgt_feats[:, -1].unsqueeze(1).expand_as(last_rollout),
                    ).mean((-2, -1))
                    best_idx = loss.argmin(dim=1)
                    best_rollout = all_rollouts[
                        torch.arange(all_rollouts.shape[0]), best_idx
                    ]

                    self._eval_horizon(best_rollout, **eval_kw, key_suffix="_best")

                self._plot_selected_samples(
                    sample_idx,
                    self.num_plots,
                    horizon_frames=horizon_frames,
                    all_feats=torch.cat([ctx_feats, tgt_feats], dim=1),
                    best_rollout=best_rollout,
                    mean_rollout=mean_rollout,
                    best_idx=best_idx,
                    labels=labels[:, ctx_start:] if labels is not None else None,
                    frame_shape=horizon_frames.shape,
                    overlap=overlap,
                    is_wide=is_wide,
                    prefix=f"{log_prefix}_c{ctx_len}_h{horizon}",
                    dataset_name=dataset_name,
                    ctx_len=ctx_len,
                )

        return self._eval_val_loss(log_prefix, dataset_name, losses)

    def _apply_head_to_rollouts(
        self,
        all_rollouts: torch.Tensor,
        task_head: torch.nn.Module,
        frame_shape: tuple[int, ...],
    ) -> torch.Tensor:
        batch_size, num_samples, num_preds = all_rollouts.shape[:3]
        feats = all_rollouts.reshape(
            batch_size * num_samples, num_preds, *all_rollouts.shape[3:]
        )
        out = self._apply_head(feats, task_head, frame_shape)
        return out.reshape(batch_size, num_samples, num_preds, *out.shape[2:])

    def _apply_pca_split(
        self,
        ctx_feats: torch.Tensor,
        tgt_feats: torch.Tensor,
        pred_feats: torch.Tensor,
        h: int,
        w: int,
        ctx_len: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        all_target = torch.cat([ctx_feats, tgt_feats], 0)
        all_pred = torch.cat([ctx_feats, pred_feats], 0)
        tgt_vis, pred_vis = feats_to_pca(all_target, all_pred, h, w)
        return tgt_vis[:ctx_len], tgt_vis[ctx_len:], pred_vis[ctx_len:]

    def _plot_feats(self, sample: dict) -> None:
        ctx_len = sample["ctx_len"]
        horizon_frames = sample["horizon_frames"]
        frames = [to_numpy_img(frame) for frame in horizon_frames[0]]
        patch_size = self.network.backbone.patch_size
        h, w = (
            horizon_frames.shape[-2] // patch_size,
            horizon_frames.shape[-1] // patch_size,
        )
        ctx_feats = sample["all_feats"][0, :ctx_len]
        tgt_feats = sample["all_feats"][0, ctx_len:]
        pca = partial(self._apply_pca_split, ctx_feats, h=h, w=w, ctx_len=ctx_len)
        num_target = tgt_feats.shape[0]

        for name in ["mean", "best"]:
            rollout = sample.get(f"{name}_rollout")
            if rollout is None:
                continue
            rollout = rollout[0]
            if rollout.shape[0] < num_target:
                ctx_disp, tgt_disp, _ = pca(tgt_feats, tgt_feats)
                pred_row = [None] * (ctx_len + num_target - rollout.shape[0]) + list(
                    pca(tgt_feats[-1:], rollout[-1:])[2]
                )
            else:
                ctx_disp, tgt_disp, pred_disp = pca(tgt_feats, rollout)
                pred_row = [None] * ctx_len + list(pred_disp)
            self._log_plot_img(
                f"{sample['prefix']}_feats_{name}",
                create_plot_from_rows(
                    [frames, list(ctx_disp) + list(tgt_disp), pred_row], len(frames)
                ),
            )

    def _plot_task(self, sample: dict, task_head: torch.nn.Module) -> None:
        task_key = TASK_HEAD_KEY[type(task_head)]
        all_oracle = self._apply_head(
            sample["all_feats"],
            task_head,
            sample["frame_shape"],
        )[0]
        frame_imgs = prepare_frame_imgs(
            sample["horizon_frames"][0],
            sample["frame_shape"],
            sample["overlap"],
            sample["is_wide"],
            sample["is_wide"] is not None,
        )

        for name in ["mean", "best"]:
            rollout = sample.get(f"{name}_rollout")
            if rollout is None:
                continue
            preds = self._apply_head(rollout, task_head, sample["frame_shape"])[0]
            if align_fn := self.head_align.get(sample["dataset_name"]):
                preds = align_fn(preds, sample["labels"][0, -len(preds) :])
            pred_task = torch.cat([all_oracle[: sample["ctx_len"]], preds], dim=0)
            gt_vis, oracle_vis, pred_vis = VIS_FNS[task_key](
                all_oracle, sample["labels"][0], pred_task
            )
            pred_row = [None] * sample["ctx_len"] + list(pred_vis[sample["ctx_len"] :])
            self._log_plot_img(
                f"{sample['prefix']}_{task_key}_{name}",
                create_plot_from_rows(
                    [frame_imgs, gt_vis, oracle_vis, pred_row], len(frame_imgs)
                ),
            )

    def _plot_rgb(self, sample: dict) -> None:
        def decode(f: torch.Tensor) -> torch.Tensor:
            return self._apply_head_to_rollouts(
                f[None, None], self.rgb_head, horizon_frames.shape
            )[0, 0]

        def to_imgs(preds: torch.Tensor) -> list:
            return [pred.permute(1, 2, 0).clamp(0, 1).cpu().numpy() for pred in preds]

        horizon_frames = sample["horizon_frames"]
        frame_imgs = [to_numpy_img(frame) for frame in horizon_frames[0]]
        best = sample.get("best_rollout")
        rollouts = {
            "mean": decode(sample["mean_rollout"][0]),
            "best": decode(best[0]) if best is not None else None,
        }
        tgt_row = to_imgs(decode(sample["all_feats"][0]))

        for name, preds in rollouts.items():
            if preds is None:
                continue
            pred_row = [None] * (len(frame_imgs) - preds.shape[0]) + to_imgs(preds)
            rows = [frame_imgs, tgt_row, pred_row]
            self._log_plot_img(
                f"{sample['prefix']}_rgb_{name}",
                create_plot_from_rows(rows, len(frame_imgs)),
            )
