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


class DeltaTok(Base):
    def __init__(
        self,
        network: torch.nn.Module,
        eval_horizons: tuple[int, ...] = (1, 3),
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        eval_oracle: bool = False,
        loss_fn: str = "log_cosh",
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
        frames, _, labels, sample_idx, overlap, is_wide = (
            preprocess_validation_batch(batch, self.trainer.datamodule.frame_size)
        )
        num_frames = frames.shape[1]
        dataset_name = self.trainer.val_dataloaders[dataloader_idx].dataset_name
        log_prefix = f"val_{dataset_name}"
        task_head = getattr(self.task_heads, dataset_name, None)

        losses = []
        for horizon in self.eval_horizons:
            ctx_len = 0 if horizon >= num_frames else 1
            pred_feats, tgt_feats, ctx_feats = self.network(
                frames, horizon=horizon
            )
            eval_feats = tgt_feats if self.eval_oracle else pred_feats
            loss = self._eval_horizon(
                eval_feats,
                tgt_feats,
                labels,
                ctx_len,
                horizon,
                task_head,
                frames.shape,
                overlap,
                is_wide,
                dataset_name,
                log_prefix,
            )
            losses.append(loss)

            all_feats = torch.cat([ctx_feats, tgt_feats], dim=1)
            self._plot_selected_samples(
                sample_idx,
                self.num_plots,
                frames=frames,
                pred_feats=pred_feats,
                all_feats=all_feats,
                eval_feats=eval_feats,
                labels=labels,
                frame_shape=frames.shape,
                overlap=overlap,
                is_wide=is_wide,
                prefix=f"{log_prefix}_c{ctx_len}_h{horizon}",
                dataset_name=dataset_name,
            )

        return self._eval_val_loss(log_prefix, dataset_name, losses)

    def _plot_feats(self, sample: dict) -> None:
        frames = sample["frames"][0]
        num_frames = len(frames)
        num_preds = sample["pred_feats"].shape[1]
        patch_size = self.network.backbone.patch_size
        h, w = frames.shape[-2] // patch_size, frames.shape[-1] // patch_size
        all_feats = sample["all_feats"][0]
        all_pred = torch.cat([all_feats[:-num_preds], sample["pred_feats"][0]], dim=0)
        tgt_vis, pred_vis = feats_to_pca(all_feats, all_pred, h, w)
        pred_row = [None] * (num_frames - num_preds) + pred_vis[-num_preds:]
        self._log_plot_img(
            f"{sample['prefix']}_feats",
            create_plot_from_rows(
                [[to_numpy_img(frame) for frame in frames], tgt_vis, pred_row],
                num_frames,
            ),
        )

    def _plot_task(self, sample: dict, task_head: torch.nn.Module) -> None:
        labels = sample["labels"]
        frame_shape, overlap, is_wide = (
            sample["frame_shape"],
            sample["overlap"],
            sample["is_wide"],
        )
        oracle = self._apply_head(sample["all_feats"], task_head, frame_shape)[0]
        preds = self._apply_head(sample["eval_feats"], task_head, frame_shape)[0]
        if align_fn := self.head_align.get(sample["dataset_name"]):
            preds = align_fn(preds, labels[0, -len(preds) :])
        num_frames = sample["frames"].shape[1]
        num_preds = len(preds)
        frame_imgs = prepare_frame_imgs(
            sample["frames"][0],
            frame_shape,
            overlap,
            is_wide,
            is_wide is not None,
        )
        task_key = TASK_HEAD_KEY[type(task_head)]
        gt_vis, oracle_vis, pred_vis = VIS_FNS[task_key](oracle, labels[0], preds)
        pred_row = [None] * (num_frames - num_preds) + pred_vis
        self._log_plot_img(
            f"{sample['prefix']}_{task_key}",
            create_plot_from_rows(
                [frame_imgs, gt_vis, oracle_vis, pred_row], num_frames
            ),
        )

    def _plot_rgb(self, sample: dict) -> None:
        frame_shape, overlap, is_wide = (
            sample["frame_shape"],
            sample["overlap"],
            sample["is_wide"],
        )
        oracle = self._apply_head(sample["all_feats"], self.rgb_head, frame_shape)[0]
        preds = self._apply_head(sample["eval_feats"], self.rgb_head, frame_shape)[0]
        num_frames = sample["frames"].shape[1]
        num_preds = len(preds)
        frame_imgs = prepare_frame_imgs(
            sample["frames"][0],
            oracle.shape[-2:],
            overlap,
            is_wide,
            is_wide is not None,
        )
        _, oracle_vis, pred_vis = VIS_FNS["rgb"](oracle, None, preds)
        pred_row = [None] * (num_frames - num_preds) + pred_vis
        self._log_plot_img(
            f"{sample['prefix']}_rgb",
            create_plot_from_rows([frame_imgs, oracle_vis, pred_row], num_frames),
        )
