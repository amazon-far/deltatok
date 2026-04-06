import torch
import torch.nn as nn

from models.predictor import Predictor
from models.world import World, causal_mask


class DINOWorld(World):
    def __init__(
        self,
        backbone: nn.Module,
        rope_axis_sizes: tuple = (20, 20, 20),
        predictor_hidden_size: int = 768,
        predictor_num_hidden_layers: int = 12,
        predictor_num_heads: int = 12,
        use_bom: bool = False,
        num_samples_train: int = 16,
        num_samples_eval: int = 20,
        layer_scale_init: float = 1e-5,
        rope_unrotated_size: int = 4,
        mlp_ratio: int = 4,
):
        super().__init__(
            **{k: v for k, v in locals().items() if k not in ("self", "__class__")},
            initializer_range=backbone.initializer_range,
        )
        self.backbone = backbone.requires_grad_(False).eval()
        self._validate_rope_sizes()

        self.predictor = Predictor(
            self.backbone.hidden_size,
            self.backbone.initializer_range,
            rope_axis_sizes,
            rope_unrotated_size,
            layer_scale_init,
            predictor_hidden_size,
            predictor_num_hidden_layers,
            predictor_num_heads,
            mlp_ratio,
        )

    def _forward_train(
        self,
        frames: torch.Tensor,
        timestamps: torch.Tensor,
        criterion: nn.Module | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(frames, timestamps, criterion)

        y = self.backbone(frames)
        batch_size, num_frames, hidden_size = y.shape[0], y.shape[1], y.shape[-1]
        tokens_per_frame, w_coords, h_coords = self._prepare_spatial_coords(
            batch_size, frames
        )

        start = 1
        num_query_frames = num_frames - start
        total_kv_len = (num_frames - 1) * tokens_per_frame
        num_queries = num_query_frames * tokens_per_frame

        kv = y[:, :-1].reshape(batch_size, total_kv_len, hidden_size)
        w_kv = w_coords.repeat(1, num_frames - 1)
        h_kv = h_coords.repeat(1, num_frames - 1)
        w_q = w_coords.repeat(1, num_query_frames)
        h_q = h_coords.repeat(1, num_query_frames)

        pos_q = (
            timestamps[:, start:].repeat_interleave(tokens_per_frame, 1),
            w_q,
            h_q,
        )
        pos_k = (
            timestamps[:, :-1].repeat_interleave(tokens_per_frame, 1),
            w_kv,
            h_kv,
        )

        if self.use_bom:
            pos_q_bom = (timestamps[:, start:], w_coords, h_coords)
            pos_k_bom = (timestamps[:, :-1], w_coords, h_coords)
            q = self._bom_queries(
                batch_size,
                num_query_frames,
                y.device,
                kv,
                pos_q_bom,
                pos_k_bom,
                y[:, start:],
                criterion,
                tokens_per_frame,
            )
        else:
            q = self._prepare_queries(batch_size, num_queries)

        q_indices = (
            torch.arange(num_queries, device=y.device) // tokens_per_frame + start
        )
        k_indices = torch.arange(total_kv_len, device=y.device) // tokens_per_frame

        y_hat = self.predictor(q, kv, causal_mask(q_indices, k_indices), pos_q, pos_k)
        y_hat = y_hat.reshape(
            batch_size, num_query_frames, tokens_per_frame, hidden_size
        )
        return y_hat, y[:, start:]

    def rollout_init(self, frames: torch.Tensor, ctx_len: int) -> dict:
        y = self.backbone(frames)
        batch_size, hidden_size = y.shape[0], y.shape[-1]
        tokens_per_frame, w_coords, h_coords = self._prepare_spatial_coords(
            batch_size, frames
        )

        kv = y[:, :ctx_len].reshape(batch_size, ctx_len * tokens_per_frame, hidden_size)
        kv = self._expand_bom(kv, self._num_samples, batch_size)

        return {"kv": kv, "y": y, "w_coords": w_coords, "h_coords": h_coords}

    def rollout_step(
        self,
        state: dict,
        tgt_frame_idx: int,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        batch_size = state["y"].shape[0]
        w_coords, h_coords = state["w_coords"], state["h_coords"]
        tokens_per_frame = w_coords.shape[1]
        num_kv_frames = state["kv"].shape[1] // tokens_per_frame

        q = self._rollout_queries(batch_size, 1, state["y"].device).repeat_interleave(
            tokens_per_frame, 1
        )

        pos_q = (
            timestamps[:, tgt_frame_idx : tgt_frame_idx + 1].repeat_interleave(
                tokens_per_frame, 1
            ),
            w_coords,
            h_coords,
        )
        pos_k = (
            timestamps[:, :num_kv_frames].repeat_interleave(tokens_per_frame, 1),
            w_coords.repeat(1, num_kv_frames),
            h_coords.repeat(1, num_kv_frames),
        )

        pos_q = self._expand_bom(pos_q, self._num_samples, batch_size)
        pos_k = self._expand_bom(pos_k, self._num_samples, batch_size)

        y_hat = self.predictor(q, state["kv"], None, pos_q, pos_k)
        state["kv"] = torch.cat([state["kv"], y_hat.detach()], 1)
        return y_hat, state["y"][:, tgt_frame_idx], state

    def _prepare_spatial_coords(
        self, batch_size: int, frames: torch.Tensor
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        patch_height, patch_width = (
            frames.shape[-2] // self.backbone.patch_size,
            frames.shape[-1] // self.backbone.patch_size,
        )
        h_coords, w_coords = torch.meshgrid(
            torch.linspace(-1, 1, patch_height, device=frames.device),
            torch.linspace(-1, 1, patch_width, device=frames.device),
            indexing="ij",
        )
        tokens_per_frame = patch_height * patch_width
        return (
            tokens_per_frame,
            w_coords.flatten().unsqueeze(0).expand(batch_size, -1),
            h_coords.flatten().unsqueeze(0).expand(batch_size, -1),
        )
