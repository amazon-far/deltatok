import torch
import torch.nn as nn

from models.deltatok import DeltaTok
from models.predictor import Predictor
from models.world import World, causal_mask


class DeltaWorld(World):
    def __init__(
        self,
        tokenizer: DeltaTok,
        rope_axis_sizes: tuple = (60,),
        predictor_hidden_size: int = 768,
        predictor_num_hidden_layers: int = 12,
        predictor_num_heads: int = 12,
        use_bom: bool = True,
        num_samples_train: int = 256,
        num_samples_eval: int = 20,
        layer_scale_init: float = 1e-5,
        rope_unrotated_size: int = 4,
        mlp_ratio: int = 4,
    ):
        super().__init__(
            **{k: v for k, v in locals().items() if k not in ("self", "__class__")},
            initializer_range=tokenizer.backbone.initializer_range,
        )
        tokenizer.requires_grad_(False).eval()
        self.backbone = tokenizer.backbone
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

        z, y, rope = self._encode_frames(frames)
        batch_size, seq_len = z.shape[0], timestamps.shape[1] - 1

        start = 1
        num_queries = timestamps.shape[1] - start

        pos_q = (timestamps[:, start:],)
        pos_k = (timestamps[:, :-1],)

        if self.use_bom:
            q = self._bom_queries(
                batch_size,
                num_queries,
                z.device,
                z[:, :seq_len],
                pos_q,
                pos_k,
                z[:, start:],
                criterion,
            )
        else:
            q = self._prepare_queries(batch_size, num_queries)

        q_indices = torch.arange(num_queries, device=z.device) + start
        k_indices = torch.arange(seq_len, device=z.device)

        z_hat = self.predictor(
            q, z[:, :seq_len], causal_mask(q_indices, k_indices), pos_q, pos_k
        )

        return z_hat, z[:, start:]

    def rollout_init(self, frames: torch.Tensor, ctx_len: int) -> dict:
        z, y, rope = self._encode_frames(frames)
        batch_size = y.shape[0]

        return {
            "kv": self._expand_bom(
                z[:, :ctx_len].detach(), self._num_samples, batch_size
            ),
            "x": self._expand_bom(y[:, ctx_len - 1], self._num_samples, batch_size),
            "y": y,
            "rope": rope,
        }

    def rollout_step(
        self,
        state: dict,
        tgt_frame_idx: int,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        batch_size = state["y"].shape[0]
        q = self._rollout_queries(batch_size, 1, state["kv"].device)

        pos_q = self._expand_bom(
            (timestamps[:, tgt_frame_idx : tgt_frame_idx + 1],),
            self._num_samples,
            batch_size,
        )
        pos_k = self._expand_bom(
            (timestamps[:, : state["kv"].shape[1]],), self._num_samples, batch_size
        )

        z_hat = self.predictor(q, state["kv"], None, pos_q, pos_k)
        y_hat = self.tokenizer.decode(z_hat, state["x"], state["rope"])

        state["kv"] = torch.cat([state["kv"], z_hat.detach()], 1)
        state["x"] = y_hat.detach()

        return y_hat, state["y"][:, tgt_frame_idx], state

    @torch.no_grad
    def _encode_frames(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        y = self.tokenizer.backbone(frames)
        x = self.tokenizer.backbone(torch.zeros_like(frames[:, :1]))[:, 0]
        rope = self.tokenizer._rope(frames)
        z = self.tokenizer.tokenize_offline(x, y, rope)
        return z.squeeze(2), y, rope
