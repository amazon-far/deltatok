import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegHead(nn.Module):
    def __init__(self, in_size: int, num_classes: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_size)
        self.head = nn.Conv2d(in_size, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.bn(x))


class DepthHead(nn.Module):
    def __init__(
        self, in_size: int, min_depth: float, max_depth: float, num_bins: int = 256
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_size)
        self.head = nn.Conv2d(in_size, num_bins, 1)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
        logits = torch.relu(self.head(self.bn(x))) + eps
        logits = logits / logits.sum(1, True)
        bins = torch.linspace(
            self.min_depth, self.max_depth, self.head.out_channels, device=x.device
        )
        return torch.einsum("bkmn,k->bmn", [logits, bins]).unsqueeze(1)


# Adapted from RAE (MIT): https://github.com/bytetriper/RAE
class RGBHead(nn.Module):
    class _Attn(nn.Module):
        def __init__(self, hidden_size: int, num_heads: int) -> None:
            super().__init__()
            self.num_heads, self.head_size = num_heads, hidden_size // num_heads
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len, hidden_size = x.shape
            q = (
                self.query(x)
                .view(batch_size, seq_len, self.num_heads, self.head_size)
                .transpose(1, 2)
            )
            k = (
                self.key(x)
                .view(batch_size, seq_len, self.num_heads, self.head_size)
                .transpose(1, 2)
            )
            v = (
                self.value(x)
                .view(batch_size, seq_len, self.num_heads, self.head_size)
                .transpose(1, 2)
            )
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
            return (
                (attn.softmax(dim=-1) @ v)
                .transpose(1, 2)
                .reshape(batch_size, seq_len, hidden_size)
            )

    class _Block(nn.Module):
        def __init__(self, hidden_size: int, num_heads: int, mlp_size: int) -> None:
            super().__init__()
            self.attention = nn.ModuleDict(
                {
                    "attention": RGBHead._Attn(hidden_size, num_heads),
                    "output": nn.ModuleDict(
                        {"dense": nn.Linear(hidden_size, hidden_size)}
                    ),
                }
            )
            self.layernorm_before = nn.LayerNorm(hidden_size, eps=1e-12)
            self.layernorm_after = nn.LayerNorm(hidden_size, eps=1e-12)
            self.intermediate = nn.ModuleDict(
                {"dense": nn.Linear(hidden_size, mlp_size)}
            )
            self.output = nn.ModuleDict({"dense": nn.Linear(mlp_size, hidden_size)})

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attention["output"]["dense"](
                self.attention["attention"](self.layernorm_before(x))
            )
            return (
                self.output["dense"](
                    F.gelu(self.intermediate["dense"](self.layernorm_after(x)))
                )
                + x
            )

    @staticmethod
    def _sincos_pos_embed(hidden_size: int, num_tokens: int) -> np.ndarray:
        grid_size = int(num_tokens**0.5)
        y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        grid = np.stack([x.flatten(), y.flatten()], -1).astype(np.float32)
        omega = 1.0 / 10000 ** (np.arange(0, hidden_size // 4) / (hidden_size // 4))
        angles = grid[:, :, None] * omega[None, None, :]
        pos_embed = np.concatenate([np.sin(angles), np.cos(angles)], -1).reshape(
            num_tokens, hidden_size
        )
        return np.concatenate([np.zeros((1, hidden_size)), pos_embed])[None]

    def __init__(
        self,
        in_size: int,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        img_mean: list[float],
        img_std: list[float],
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_size: int = 4096,
        patch: int = 16,
        img_size: int = 512,
    ) -> None:
        super().__init__()
        num_patches = (img_size // patch) ** 2
        self.patch = patch
        self.decoder_embed = nn.Linear(in_size, hidden_size)
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.from_numpy(self._sincos_pos_embed(hidden_size, num_patches)).float(),
            requires_grad=False,
        )
        self.decoder_layers = nn.ModuleList(
            [self._Block(hidden_size, num_heads, mlp_size) for _ in range(depth)]
        )
        self.decoder_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder_pred = nn.Linear(hidden_size, patch**2 * 3)
        self.register_buffer("norm_weight", norm_weight, persistent=False)
        self.register_buffer("norm_bias", norm_bias, persistent=False)
        self.register_buffer(
            "img_mean",
            torch.tensor(img_mean, dtype=norm_weight.dtype).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor(img_std, dtype=norm_weight.dtype).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = (x - self.norm_bias) / self.norm_weight
        x = torch.cat(
            [
                self.trainable_cls_token.expand(batch_size, -1, -1),
                self.decoder_embed(x),
            ],
            1,
        )
        pos_embed = self.decoder_pos_embed
        original_grid = int((pos_embed.shape[1] - 1) ** 0.5)
        if original_grid != h or original_grid != w:
            patch_pos = (
                pos_embed[:, 1:]
                .view(1, original_grid, original_grid, -1)
                .permute(0, 3, 1, 2)
            )
            patch_pos = F.interpolate(patch_pos, (h, w), mode="bicubic")
            pos_embed = torch.cat(
                [pos_embed[:, :1], patch_pos.permute(0, 2, 3, 1).flatten(1, 2)], 1
            )
        x = x + pos_embed
        for blk in self.decoder_layers:
            x = blk(x)
        x = self.decoder_pred(self.decoder_norm(x))[:, 1:]
        x = (
            x.view(batch_size, h, w, self.patch, self.patch, 3)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch_size, 3, h * self.patch, w * self.patch)
        )
        return x * self.img_std + self.img_mean
