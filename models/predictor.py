import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTLayerScale,
    DINOv3ViTMLP,
)

DINOV3_TEMPLATE = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _apply_rope_axis(x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    half_size = x.shape[-1] // 2
    inv_freq = 1.0 / (10.0 ** torch.linspace(-2, 2, half_size, device=x.device))
    x_even, x_odd = x[..., :half_size], x[..., half_size : 2 * half_size]
    angle = (2 * torch.pi) * position * inv_freq.view(1, 1, 1, half_size)
    sin_vals, cos_vals = angle.sin(), angle.cos()
    return torch.cat(
        [x_even * cos_vals - x_odd * sin_vals, x_even * sin_vals + x_odd * cos_vals], -1
    )


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_q: list[torch.Tensor],
    pos_k: list[torch.Tensor],
    rope_axis_sizes: list[int],
    rope_unrotated_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rotated, k_rotated, offset = [], [], 0
    for i, axis_size in enumerate(rope_axis_sizes):
        q_rotated.append(
            _apply_rope_axis(q[..., offset : offset + axis_size], pos_q[i])
        )
        k_rotated.append(
            _apply_rope_axis(k[..., offset : offset + axis_size], pos_k[i])
        )
        offset += axis_size
    q_rotated.append(q[..., offset : offset + rope_unrotated_size])
    k_rotated.append(k[..., offset : offset + rope_unrotated_size])
    return torch.cat(q_rotated, -1), torch.cat(k_rotated, -1)


def _reshape_heads(
    x: torch.Tensor, seq_len: int, batch_size: int, num_heads: int, head_size: int
) -> torch.Tensor:
    return x.reshape(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)


class Predictor(nn.Module):
    class _CrossAttn(nn.Module):
        def __init__(
            self,
            backbone_hidden_size: int,
            rope_axis_sizes: list[int],
            rope_unrotated_size: int,
            predictor_hidden_size: int,
            num_attention_heads: int,
        ) -> None:
            super().__init__()
            self.rope_axis_sizes = rope_axis_sizes
            self.rope_unrotated_size = rope_unrotated_size
            self.predictor_hidden_size = predictor_hidden_size
            self.num_heads = num_attention_heads
            self.head_size = predictor_hidden_size // num_attention_heads
            self.q_proj = nn.Linear(predictor_hidden_size, predictor_hidden_size)
            self.kv_proj = nn.Linear(backbone_hidden_size, 2 * predictor_hidden_size)
            self.out_proj = nn.Linear(predictor_hidden_size, predictor_hidden_size)

        def forward(
            self,
            q_input: torch.Tensor,
            kv_input: torch.Tensor,
            attn_mask: torch.Tensor | None,
            pos_q: tuple[torch.Tensor, ...],
            pos_k: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            batch_size, query_len, key_len = (
                q_input.shape[0],
                q_input.shape[1],
                kv_input.shape[1],
            )

            q = _reshape_heads(
                self.q_proj(q_input),
                query_len,
                batch_size,
                self.num_heads,
                self.head_size,
            )
            k, v = self.kv_proj(kv_input).chunk(2, dim=-1)
            k = _reshape_heads(k, key_len, batch_size, self.num_heads, self.head_size)
            v = _reshape_heads(v, key_len, batch_size, self.num_heads, self.head_size)

            def _expand_pos(pos_vals, seq_len):
                return (
                    pos_vals[:, None, :]
                    .unsqueeze(-1)
                    .expand(batch_size, self.num_heads, seq_len, 1)
                )

            q, k = _apply_rope(
                q,
                k,
                [_expand_pos(pos_vals, query_len) for pos_vals in pos_q],
                [_expand_pos(pos_vals, key_len) for pos_vals in pos_k],
                self.rope_axis_sizes,
                self.rope_unrotated_size,
            )
            out = F.scaled_dot_product_attention(q, k, v, attn_mask)
            return self.out_proj(
                out.permute(0, 2, 1, 3).reshape(
                    batch_size, query_len, self.predictor_hidden_size
                )
            )

    class _Block(nn.Module):
        def __init__(
            self,
            cfg: AutoConfig,
            backbone_hidden_size: int,
            rope_axis_sizes: list[int],
            rope_unrotated_size: int,
            predictor_hidden_size: int,
            num_attention_heads: int,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(predictor_hidden_size)
            self.attention = Predictor._CrossAttn(
                backbone_hidden_size,
                rope_axis_sizes,
                rope_unrotated_size,
                predictor_hidden_size,
                num_attention_heads,
            )
            self.layer_scale1 = DINOv3ViTLayerScale(cfg)
            self.norm2 = nn.LayerNorm(predictor_hidden_size)
            self.mlp = DINOv3ViTMLP(cfg)
            self.layer_scale2 = DINOv3ViTLayerScale(cfg)

        def forward(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
            mask: torch.Tensor | None,
            pos_q: tuple[torch.Tensor, ...],
            pos_k: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            q = q + self.layer_scale1(
                self.attention(self.norm1(q), kv, mask, pos_q, pos_k)
            )
            q = q + self.layer_scale2(self.mlp(self.norm2(q)))
            return q

    def __init__(
        self,
        backbone_hidden_size: int,
        initializer_range: float,
        rope_axis_sizes: tuple[int, ...],
        rope_unrotated_size: int,
        layer_scale_init: float,
        predictor_hidden_size: int,
        predictor_num_hidden_layers: int,
        predictor_num_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        cfg = AutoConfig.from_pretrained(DINOV3_TEMPLATE)
        cfg.hidden_size = predictor_hidden_size
        cfg.intermediate_size = predictor_hidden_size * mlp_ratio

        self.blocks = nn.ModuleList()
        for _ in range(predictor_num_hidden_layers):
            blk = self._Block(
                cfg,
                backbone_hidden_size,
                list(rope_axis_sizes),
                rope_unrotated_size,
                predictor_hidden_size,
                predictor_num_heads,
            )
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=initializer_range)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            blk.layer_scale1.lambda1.data.fill_(layer_scale_init)
            blk.layer_scale2.lambda1.data.fill_(layer_scale_init)
            self.blocks.append(blk)

        self.norm = nn.LayerNorm(predictor_hidden_size)
        self.head = nn.Linear(predictor_hidden_size, backbone_hidden_size)
        nn.init.trunc_normal_(self.head.weight, std=initializer_range)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: torch.Tensor | None,
        pos_q: tuple[torch.Tensor, ...],
        pos_k: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        for blk in self.blocks:
            q = blk(q, kv, mask, pos_q, pos_k)
        return self.head(self.norm(q))
