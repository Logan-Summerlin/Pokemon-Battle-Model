"""Simple MLP and GRU baseline models for behavior cloning.

Phase 3 baseline: 2-layer MLP or small GRU over flattened observations.
Trained with masked cross-entropy loss over legal actions.

Input: Flattened tensor from tensorizer (own_team + opp_team + field + context)
Output: Logits over NUM_ACTIONS, masked by legal action mask
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.tensorizer import (
    CONTEXT_FEATURE_DIM,
    FIELD_FEATURE_DIM,
    POKEMON_FEATURE_DIM,
)
from src.data.observation import MAX_TEAM_SIZE
from src.environment.action_space import NUM_ACTIONS


# Total flattened input dimension
FLAT_INPUT_DIM = (
    MAX_TEAM_SIZE * POKEMON_FEATURE_DIM  # own team: 6 * 30 = 180
    + MAX_TEAM_SIZE * POKEMON_FEATURE_DIM  # opp team: 6 * 30 = 180
    + FIELD_FEATURE_DIM  # field: 18
    + CONTEXT_FEATURE_DIM  # context: 6
)  # Total: 384


class BaselineMLP(nn.Module):
    """2-layer MLP baseline for behavior cloning.

    Flattens the observation tensors, passes through hidden layers,
    and outputs logits over the action space. Legal action masking
    is applied before softmax during loss computation.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        input_dim: int = FLAT_INPUT_DIM,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_dim = input_dim
        self.num_actions = num_actions

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, num_actions)

    def forward(
        self,
        own_team: torch.Tensor,
        opponent_team: torch.Tensor,
        field: torch.Tensor,
        context: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            own_team: (batch, 6, POKEMON_FEATURE_DIM) or (batch, seq, 6, POKEMON_FEATURE_DIM)
            opponent_team: same shape as own_team
            field: (batch, FIELD_FEATURE_DIM) or (batch, seq, FIELD_FEATURE_DIM)
            context: (batch, CONTEXT_FEATURE_DIM) or (batch, seq, CONTEXT_FEATURE_DIM)
            legal_mask: (batch, NUM_ACTIONS) or (batch, seq, NUM_ACTIONS), 1=legal 0=illegal

        Returns:
            Logits of shape (batch, NUM_ACTIONS) or (batch, seq, NUM_ACTIONS).
            Illegal actions are masked to -inf if legal_mask is provided.
        """
        # Flatten team tensors
        batch_shape = own_team.shape[:-2]  # Everything before (6, feature_dim)
        own_flat = own_team.reshape(*batch_shape, -1)
        opp_flat = opponent_team.reshape(*batch_shape, -1)

        # Concatenate all features
        x = torch.cat([own_flat, opp_flat, field, context], dim=-1)

        # MLP
        x = self.backbone(x)
        logits = self.head(x)

        # Apply legal action mask
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, float("-inf"))

        return logits


class BaselineGRU(nn.Module):
    """Small GRU baseline for behavior cloning.

    Processes the flattened observation through a GRU to capture
    sequential context from the battle history, then outputs
    action logits.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        input_dim: int = FLAT_INPUT_DIM,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_actions)

    def forward(
        self,
        own_team: torch.Tensor,
        opponent_team: torch.Tensor,
        field: torch.Tensor,
        context: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        seq_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for sequence input.

        Args:
            own_team: (batch, seq, 6, POKEMON_FEATURE_DIM)
            opponent_team: (batch, seq, 6, POKEMON_FEATURE_DIM)
            field: (batch, seq, FIELD_FEATURE_DIM)
            context: (batch, seq, CONTEXT_FEATURE_DIM)
            legal_mask: (batch, seq, NUM_ACTIONS)
            seq_len: (batch,) actual sequence lengths for packing

        Returns:
            Logits of shape (batch, seq, NUM_ACTIONS).
        """
        batch_size = own_team.shape[0]
        seq_length = own_team.shape[1]

        # Flatten team tensors
        own_flat = own_team.reshape(batch_size, seq_length, -1)
        opp_flat = opponent_team.reshape(batch_size, seq_length, -1)

        # Concatenate
        x = torch.cat([own_flat, opp_flat, field, context], dim=-1)

        # Project to hidden dim
        x = F.relu(self.input_proj(x))

        # GRU
        if seq_len is not None:
            # Pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_len.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.gru(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq_length,
            )
        else:
            x, _ = self.gru(x)

        x = self.dropout(x)
        logits = self.head(x)

        # Apply legal action mask
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, float("-inf"))

        return logits


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Cross-entropy loss masked to only consider legal actions.

    Args:
        logits: (batch, num_actions) or (batch, seq, num_actions) raw logits
        targets: (batch,) or (batch, seq) target action indices
        legal_mask: (batch, num_actions) or (batch, seq, num_actions)
        ignore_index: target value to ignore (e.g., padding)

    Returns:
        Scalar loss tensor.
    """
    # Mask illegal actions to -inf before computing CE
    masked_logits = logits.masked_fill(legal_mask == 0, float("-inf"))

    # Flatten if needed for CE loss
    if masked_logits.dim() == 3:
        batch, seq, n_act = masked_logits.shape
        masked_logits = masked_logits.reshape(-1, n_act)
        targets = targets.reshape(-1)

    # Filter out positions where all logits are -inf (fully masked padding)
    # to avoid NaN losses, in addition to ignore_index filtering.
    valid = targets != ignore_index
    if valid.any():
        return F.cross_entropy(
            masked_logits[valid], targets[valid], ignore_index=ignore_index
        )
    return torch.tensor(0.0, device=logits.device, requires_grad=True)


def create_baseline_model(
    architecture: str = "mlp",
    hidden_dims: list[int] | None = None,
    gru_hidden_dim: int = 256,
    gru_num_layers: int = 1,
    dropout: float = 0.2,
) -> nn.Module:
    """Factory function to create a baseline model.

    Args:
        architecture: "mlp" or "gru"
        hidden_dims: Hidden layer dimensions for MLP
        gru_hidden_dim: Hidden dimension for GRU
        gru_num_layers: Number of GRU layers
        dropout: Dropout rate

    Returns:
        A BaselineMLP or BaselineGRU model.
    """
    if architecture == "mlp":
        return BaselineMLP(hidden_dims=hidden_dims, dropout=dropout)
    elif architecture == "gru":
        return BaselineGRU(
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
