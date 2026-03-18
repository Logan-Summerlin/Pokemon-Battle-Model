"""Muon optimizer: momentum + Newton-Schulz orthogonalization.

Based on Keller Jordan's NanoGPT speed record work (2024) and
"Muon is Scalable for LLM Training" (arXiv 2502.16982, Feb 2025).

Muon applies Nesterov momentum followed by matrix orthogonalization via
Newton-Schulz iteration to the update. It can be interpreted as steepest
descent under the spectral norm.

Usage:
    Split parameters into two groups:
    - Muon for 2D+ weight matrices (Linear layers, attention projections)
    - AdamW for embeddings, biases, LayerNorm (1D params / special layers)

    muon_params, adam_params = split_params_for_muon(model)
    muon_opt = Muon(muon_params, lr=0.02, momentum=0.95)
    adam_opt = torch.optim.AdamW(adam_params, lr=1e-4)
    combined = CombinedOptimizer(muon_opt, adam_opt)
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    """Muon optimizer: Nesterov momentum + Newton-Schulz orthogonalization.

    For 2D+ weight matrices, the gradient update is orthogonalized via
    Newton-Schulz iteration, which acts as steepest descent under the
    spectral norm. For 1D parameters (biases), plain momentum is used.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 0.02).
        momentum: Momentum factor (default: 0.95).
        nesterov: Use Nesterov momentum (default: True).
        ns_steps: Newton-Schulz iteration steps (default: 5).
        weight_decay: Decoupled weight decay (default: 0.0).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
                     (not typically used with Muon).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                # Decoupled weight decay
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # Momentum buffer
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                # Nesterov update
                if nesterov:
                    update = g + momentum * buf
                else:
                    update = buf.clone()

                # Newton-Schulz orthogonalization for 2D+ weight matrices
                if update.dim() >= 2:
                    update = _newton_schulz(update, steps=ns_steps)

                p.add_(update, alpha=-lr)

        return loss


def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate matrix orthogonalization via Newton-Schulz iteration.

    Computes an approximate orthogonal matrix from the gradient matrix G
    using the cubic Newton-Schulz iteration with optimized coefficients.

    Args:
        G: Gradient tensor (at least 2D).
        steps: Number of iteration steps (default: 5).

    Returns:
        Orthogonalized gradient tensor with the same shape as G.
    """
    # Optimized coefficients for cubic Newton-Schulz iteration
    a, b, c = (3.4445, -4.7750, 2.0315)

    shape = G.shape
    if G.dim() > 2:
        G = G.reshape(G.shape[0], -1)

    # Normalize to unit Frobenius norm for numerical stability
    X = G / (G.norm() + 1e-7)

    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))

    return X.reshape(shape)


def split_params_for_muon(
    model: torch.nn.Module,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split model parameters into Muon-eligible and AdamW-eligible groups.

    Muon is applied to 2D+ weight matrices (Linear, attention projections).
    AdamW handles embeddings, biases, LayerNorm, and other 1D parameters.

    Args:
        model: The model whose parameters to split.

    Returns:
        (muon_params, adam_params): Lists of parameter dicts suitable for
        passing to Muon() and AdamW() constructors.
    """
    muon_params = []
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # AdamW for: embeddings, norms, biases, 1D params
        is_embedding = "emb" in name
        is_norm = "norm" in name
        is_bias = name.endswith(".bias")
        is_1d = p.dim() < 2

        if is_embedding or is_norm or is_bias or is_1d:
            adam_params.append(p)
        else:
            muon_params.append(p)

    return muon_params, adam_params


class CombinedOptimizer:
    """Wraps two optimizers (Muon + AdamW) with a unified interface.

    Provides step(), zero_grad(), state_dict(), and load_state_dict()
    that delegate to both underlying optimizers.
    """

    def __init__(self, muon_optimizer: Muon, adam_optimizer: Optimizer):
        self.muon = muon_optimizer
        self.adam = adam_optimizer
        self.optimizers = [self.muon, self.adam]

    @property
    def param_groups(self) -> list:
        """Combined param groups from both optimizers."""
        return self.muon.param_groups + self.adam.param_groups

    def step(self, closure=None):
        """Step both optimizers."""
        self.muon.step(closure)
        self.adam.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients in both optimizers."""
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        """Return combined state dict."""
        return {
            "muon": self.muon.state_dict(),
            "adam": self.adam.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load combined state dict."""
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])
