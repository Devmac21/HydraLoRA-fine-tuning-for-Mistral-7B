"""
HydraLoRA: Multi-Expert LoRA with Sparse Routing
Paper-faithful implementation for parameter-efficient fine-tuning of LLMs.

Supports both soft routing (all experts) and top-k sparse routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HydraLoRA(nn.Module):
    """
    HydraLoRA layer: Multiple LoRA experts with a router.
    Each expert has its own A and B matrices; router selects top-k experts.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        num_experts: int = 8,
        alpha: int = 32,
        dropout: float = 0.05,
        top_k: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k  # None = soft routing (all experts), int = top-k sparse
        self.scaling = alpha / r

        self.lora_A = nn.ModuleList([
            nn.Linear(in_features, r, bias=False)
            for _ in range(num_experts)
        ])
        self.lora_B = nn.ModuleList([
            nn.Linear(r, out_features, bias=False)
            for _ in range(num_experts)
        ])

        self.router = nn.Linear(in_features, num_experts)
        self.dropout = nn.Dropout(dropout)

        for A, B in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(A.weight, a=5 ** 0.5)
            nn.init.zeros_(B.weight)

        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stable router logits
        logits = self.router(x)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        logits = torch.clamp(logits, -20, 20)

        if self.top_k is None:
            # Soft routing: use all experts
            gates = F.softmax(logits, dim=-1)
            expert_outs = []
            for A, B in zip(self.lora_A, self.lora_B):
                expert_outs.append(B(self.dropout(A(x))) * self.scaling)
            expert_outs = torch.stack(expert_outs, dim=-2)
            return torch.sum(gates.unsqueeze(-1) * expert_outs, dim=-2)
        else:
            # Top-k sparse routing
            topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
            gates = F.softmax(topk_vals, dim=-1)

            expert_outs = []
            for A, B in zip(self.lora_A, self.lora_B):
                expert_outs.append(B(self.dropout(A(x))) * self.scaling)
            expert_outs = torch.stack(expert_outs, dim=-2)

            out = torch.zeros_like(expert_outs[..., 0, :])
            for i in range(self.top_k):
                idx = topk_idx[..., i]
                gate = gates[..., i].unsqueeze(-1)
                selected = torch.gather(
                    expert_outs,
                    dim=-2,
                    index=idx.unsqueeze(-1).unsqueeze(-1).expand(
                        -1, -1, 1, expert_outs.size(-1)
                    ),
                ).squeeze(-2)
                out += gate * selected
            return out


class HydraLoRALinear(nn.Module):
    """Wraps a base Linear layer with HydraLoRA adapter."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        num_experts: int = 8,
        alpha: int = 32,
        dropout: float = 0.05,
        top_k: Optional[int] = 2,
    ):
        super().__init__()
        self.base = base_layer
        self.hydra = HydraLoRA(
            base_layer.in_features,
            base_layer.out_features,
            r=r,
            num_experts=num_experts,
            alpha=alpha,
            dropout=dropout,
            top_k=top_k,
        )

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.hydra.to(device=device, dtype=dtype)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.hydra(x)


def apply_hydralora(
    model: nn.Module,
    target_modules: tuple = ("q_proj", "v_proj"),
    r: int = 8,
    num_experts: int = 8,
    alpha: int = 32,
    dropout: float = 0.05,
    top_k: Optional[int] = 2,
) -> None:
    """
    Inject HydraLoRA adapters into the model's attention layers.

    Args:
        model: The transformer model (e.g., Mistral, LLaMA)
        target_modules: Tuple of layer name suffixes to target
        r: LoRA rank
        num_experts: Number of experts per layer
        alpha: LoRA scaling (alpha / r)
        dropout: Dropout in expert forward pass
        top_k: None for soft routing, int for top-k sparse (e.g., 2)
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(target_modules):
            parent = model
            *path, attr = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            original = getattr(parent, attr)
            setattr(
                parent,
                attr,
                HydraLoRALinear(
                    original,
                    r=r,
                    num_experts=num_experts,
                    alpha=alpha,
                    dropout=dropout,
                    top_k=top_k,
                ),
            )
