from abc import ABC, abstractmethod
from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from torch.func import functional_call


def causal_mask(q_indices: torch.Tensor, k_indices: torch.Tensor) -> torch.Tensor:
    mask = (k_indices[None, :] >= q_indices[:, None])[None, None, :, :]
    return torch.where(mask, float("-inf"), 0.0)


class World(nn.Module, ABC):
    def __init__(self, **kw: Any) -> None:
        super().__init__()
        for key, value in kw.items():
            setattr(self, key, value)
        if not self.use_bom:
            self.q_embed = nn.Embedding(1, self.predictor_hidden_size)
            nn.init.trunc_normal_(self.q_embed.weight, std=self.initializer_range)

    @property
    def _num_samples(self) -> int:
        return self.num_samples_eval if self.use_bom else 1

    def forward(
        self,
        frames: torch.Tensor,
        timestamps: torch.Tensor,
        criterion: nn.Module | None = None,
        ctx_len: int | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if self.training:
            return self._forward_train(frames, timestamps, criterion)
        return self._forward_eval(frames, timestamps, ctx_len)

    @abstractmethod
    def _forward_train(
        self,
        frames: torch.Tensor,
        timestamps: torch.Tensor,
        criterion: nn.Module | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def rollout_init(self, frames: torch.Tensor, ctx_len: int) -> dict: ...

    @abstractmethod
    def rollout_step(
        self, state: dict, tgt_frame_idx: int, timestamps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]: ...

    @torch.compiler.disable
    def _forward_eval(
        self, frames: torch.Tensor, timestamps: torch.Tensor, ctx_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.rollout_init(frames, ctx_len)

        preds = []
        tgts = []
        for t in range(frames.shape[1] - ctx_len):
            y_hat, y, state = self.rollout_step(state, ctx_len + t, timestamps)
            preds.append(y_hat)
            tgts.append(y)

        preds = torch.stack(preds, 1)
        tgts = torch.stack(tgts, 1)

        batch_size = state["y"].shape[0]
        preds = preds.reshape(batch_size, self._num_samples, *preds.shape[1:])
        return preds, tgts, state["y"][:, :ctx_len]

    def _validate_rope_sizes(self) -> None:
        head_size = self.predictor_hidden_size // self.predictor_num_heads
        rope_total = sum(self.rope_axis_sizes) + self.rope_unrotated_size
        assert (
            rope_total == head_size
        ), f"RoPE sizes must sum to head_size: {self.rope_axis_sizes} + {self.rope_unrotated_size} = {rope_total}, but head_size = {head_size}"

    def _validate_inputs(
        self,
        frames: torch.Tensor,
        timestamps: torch.Tensor,
        criterion: nn.Module | None,
    ) -> None:
        assert (
            frames.ndim == 5
        ), f"frames should be 5D (batch, time, C, H, W), got {frames.shape}"
        assert (
            timestamps.ndim == 2
        ), f"timestamps should be 2D (batch, time), got {timestamps.shape}"
        assert frames.shape[0] == timestamps.shape[0], "batch size mismatch"
        assert frames.shape[1] == timestamps.shape[1], "num_frames mismatch"
        if self.use_bom:
            assert criterion is not None, "criterion required for best-of-many"

    def _sample_queries(
        self, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        return (
            torch.randn(*shape, self.predictor_hidden_size, device=device)
            * self.initializer_range
        )

    def _prepare_queries(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return self.q_embed.weight.expand(batch_size, seq_len, -1)

    def _rollout_queries(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        if self.use_bom:
            q = self._sample_queries(
                (batch_size, self.num_samples_eval, seq_len), device
            )
            return q.reshape(batch_size * self.num_samples_eval, seq_len, -1)
        return self._prepare_queries(batch_size, seq_len)

    def _bom_queries(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        kv: torch.Tensor,
        pos_q: tuple[torch.Tensor, ...],
        pos_k: tuple[torch.Tensor, ...],
        tgts: torch.Tensor,
        criterion: nn.Module,
        tokens_per_frame: int = 1,
    ) -> torch.Tensor:
        q_samples = self._sample_queries(
            (batch_size, seq_len, self.num_samples_train), device
        )
        q_all = q_samples.flatten(1, 2).repeat_interleave(tokens_per_frame, 1)

        pos_q_bom = (
            pos_q[0]
            .repeat_interleave(self.num_samples_train, 1)
            .repeat_interleave(tokens_per_frame, 1),
        ) + tuple(
            pq.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, seq_len, self.num_samples_train, -1)
            .flatten(1, 3)
            for pq in pos_q[1:]
        )
        pos_k_bom = (pos_k[0].repeat_interleave(tokens_per_frame, 1),) + tuple(
            pk.unsqueeze(1).expand(-1, seq_len, -1).flatten(1, 2) for pk in pos_k[1:]
        )

        total_len = seq_len * tokens_per_frame
        start = 1
        q_indices = (
            torch.arange(total_len, device=device).repeat_interleave(
                self.num_samples_train
            )
            // tokens_per_frame
            + start
        )
        k_indices = torch.arange(total_len, device=device) // tokens_per_frame

        preds = functional_call(
            self.predictor,
            self._get_predictor_state(),
            (q_all, kv, causal_mask(q_indices, k_indices), pos_q_bom, pos_k_bom),
        ).reshape(batch_size, seq_len, self.num_samples_train, tokens_per_frame, -1)

        tgts = tgts.reshape(batch_size, seq_len, -1, tgts.shape[-1])
        losses = (
            criterion(preds, tgts.unsqueeze(2).expand_as(preds)).flatten(3).mean(-1)
        )
        batch_idx = torch.arange(batch_size, device=device)[:, None]
        seq_idx = torch.arange(seq_len, device=device)[None, :]
        return q_samples[batch_idx, seq_idx, losses.argmin(-1)].repeat_interleave(
            tokens_per_frame, 1
        )

    def _expand_bom(
        self,
        x: torch.Tensor | tuple[torch.Tensor, ...],
        num_samples: int,
        batch_size: int,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(x, tuple):
            return tuple(
                self._expand_bom(tensor, num_samples, batch_size) for tensor in x
            )
        return (
            x.unsqueeze(1)
            .expand(-1, num_samples, *[-1] * (x.ndim - 1))
            .reshape(batch_size * num_samples, *x.shape[1:])
        )

    def _get_predictor_state(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach()
            for k, v in chain(
                self.predictor.named_parameters(), self.predictor.named_buffers()
            )
        }
