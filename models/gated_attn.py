from types import MethodType

import torch
import torch.nn as nn


def _patch_dinov3_gated_attention(attn: torch.nn.Module) -> None:
    from torch.nn.functional import scaled_dot_product_attention
    from transformers.models.dinov3_vit.modeling_dinov3_vit import apply_rotary_pos_emb

    if getattr(attn, "_gated_attn_patched", False):
        return

    attn.attn_gate = nn.Linear(attn.config.hidden_size, attn.num_heads, bias=True)
    nn.init.trunc_normal_(attn.attn_gate.weight, std=attn.config.initializer_range)
    nn.init.zeros_(attn.attn_gate.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kw,
    ):
        batch_size, patches, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if hasattr(self, "q_norm"):
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, scale=self.scaling
        )

        gate = self.attn_gate(hidden_states)
        gate = gate.transpose(1, 2).unsqueeze(-1).sigmoid()
        attn_output = attn_output * gate

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, patches, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    attn.forward = MethodType(forward, attn)
    attn._gated_attn_patched = True


def enable_gated_attn(module: torch.nn.Module) -> None:
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTAttention

    for m in module.modules():
        if isinstance(m, DINOv3ViTAttention):
            _patch_dinov3_gated_attention(m)
