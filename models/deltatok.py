from itertools import chain

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTLayer,
    DINOv3ViTRopePositionEmbedding,
)

from models.gated_attn import enable_gated_attn
from models.predictor import DINOV3_TEMPLATE
from models.qk_norm import enable_dinov3_qk_norm
from training.base import load_sd


class DeltaTok(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_hidden_layers=12,
        use_delta=True,
        use_decoder_head: bool | None = None,
        use_decoder_xy_embed: bool | None = None,
        layer_scale_init=1e-5,
        use_qk_norm=True,
        use_gated_attn=True,
        use_swiglu=True,
        use_rope_aug=True,
        ckpt_path: str | None = None,
    ):
        super().__init__()
        ckpt_path = None if ckpt_path == "None" else ckpt_path
        self.use_delta = use_delta
        self.use_decoder_head = (
            use_decoder_head if use_decoder_head is not None else not use_delta
        )
        self.use_decoder_xy_embed = (
            use_decoder_xy_embed if use_decoder_xy_embed is not None else not use_delta
        )
        self.backbone = backbone.requires_grad_(False).eval()

        cfg = AutoConfig.from_pretrained(DINOV3_TEMPLATE)
        cfg._attn_implementation = "sdpa"

        cfg.hidden_size = self.backbone.hidden_size
        cfg.num_attention_heads = self.backbone.num_heads
        cfg.patch_size = self.backbone.patch_size

        if use_swiglu:
            cfg.use_gated_mlp = True
            cfg.intermediate_size = max(1, (2 * int(cfg.intermediate_size)) // 3)
            cfg.hidden_act = "silu"

        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(cfg)
        if not use_rope_aug:
            self.rope_embeddings.eval()

        self.z_embed = nn.Embedding(1, self.backbone.hidden_size)
        nn.init.trunc_normal_(self.z_embed.weight, std=cfg.initializer_range)
        num_xy = 2 if (use_delta or self.use_decoder_xy_embed) else 1
        self.xy_embed = nn.Embedding(num_xy, self.backbone.hidden_size)
        nn.init.trunc_normal_(self.xy_embed.weight, std=cfg.initializer_range)

        self.encoder_blocks = nn.ModuleList(
            [DINOv3ViTLayer(cfg) for _ in range(num_hidden_layers)]
        )
        self.decoder_blocks = nn.ModuleList(
            [DINOv3ViTLayer(cfg) for _ in range(num_hidden_layers)]
        )
        for blk in chain(self.encoder_blocks, self.decoder_blocks):
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=cfg.initializer_range)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            blk.layer_scale1.lambda1.data.fill_(layer_scale_init)
            blk.layer_scale2.lambda1.data.fill_(layer_scale_init)
            if use_qk_norm:
                enable_dinov3_qk_norm(blk)
            if use_gated_attn:
                enable_gated_attn(blk)

        self.norm = nn.LayerNorm(self.backbone.hidden_size, cfg.layer_norm_eps)

        if self.use_decoder_head:
            self.decoder_norm = nn.LayerNorm(
                self.backbone.hidden_size, cfg.layer_norm_eps
            )
            self.decoder_head = nn.Linear(
                self.backbone.hidden_size, self.backbone.hidden_size
            )
            nn.init.trunc_normal_(self.decoder_head.weight, std=cfg.initializer_range)
            nn.init.zeros_(self.decoder_head.bias)

        if ckpt_path:
            load_sd(self, torch.load(ckpt_path))

    def forward(
        self, frames: torch.Tensor, *_, horizon: int | None = None
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if self.training:
            return self._forward_train(frames)
        return self._forward_eval(frames, horizon or frames.shape[1])

    def encode(
        self, x: torch.Tensor, y: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        z = self.z_embed.weight[None].repeat(y.shape[0], 1, 1)
        if self.use_delta:
            hidden = torch.cat(
                (
                    z,
                    x + self.xy_embed.weight[0],
                    y + self.xy_embed.weight[1],
                ),
                1,
            )
            rope = (
                torch.cat((rope[0], rope[0]), -2),
                torch.cat((rope[1], rope[1]), -2),
            )
        else:
            hidden = torch.cat((z, y + self.xy_embed.weight[-1]), 1)
        for blk in self.encoder_blocks:
            hidden = blk(hidden, position_embeddings=rope)
        z = hidden[:, :1]
        return self.norm(z)

    def decode(
        self, z: torch.Tensor, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if not self.use_delta:
            if self.use_decoder_xy_embed:
                x = self.xy_embed.weight[0][None, None].repeat(
                    x.shape[0], x.shape[1], 1
                )
            else:
                x = x.new_zeros(x.shape)
        hidden = torch.cat((z, x), 1)
        for blk in self.decoder_blocks:
            hidden = blk(hidden, position_embeddings=rope)
        y_hat = hidden[:, 1:]
        if self.use_decoder_head:
            y_hat = self.decoder_head(self.decoder_norm(y_hat))
        return y_hat

    def tokenize_offline(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = torch.cat((x.unsqueeze(1), y[:, :-1]), 1)
        z = self.encode(x.flatten(0, 1), y.flatten(0, 1), rope)
        return z.unflatten(0, (y.shape[0], -1))

    def _rope(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rope_embeddings(torch.zeros_like(frames[:1, 0], dtype=torch.float))

    def _forward_train(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self._rope(frames)
        y = self.backbone(frames)
        z = self.encode(y[:, -2], y[:, -1], rope)
        y_hat = self.decode(z, y[:, -2], rope)
        return y_hat, y[:, -1]

    def rollout_init(self, frames: torch.Tensor, horizon: int) -> dict:
        y = self.backbone(frames)
        rope = self._rope(frames)
        if horizon == y.shape[1]:  # no context: start from black frame
            x = self.backbone(torch.zeros_like(frames[:, :1]))[:, 0]
        else:
            x = y[:, -horizon - 1]
        return {"y": y, "rope": rope, "x": x}

    def rollout_step(
        self, state: dict, tgt_frame_idx: int, _: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        y = state["y"][:, tgt_frame_idx]
        z = self.encode(state["x"], y, state["rope"])
        y_hat = self.decode(z, state["x"], state["rope"])
        state = {**state, "x": y_hat.detach()}
        return y_hat, y, state

    @torch.compiler.disable
    def _forward_eval(
        self, frames: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.rollout_init(frames, horizon)
        preds, tgts = [], []
        for t in range(horizon):
            y_hat, y, state = self.rollout_step(
                state, frames.shape[1] - horizon + t, None
            )
            preds.append(y_hat)
            tgts.append(y)
        preds = torch.stack(preds, 1)
        tgts = torch.stack(tgts, 1)
        return preds, tgts, state["y"][:, : frames.shape[1] - horizon]
