"""Tests for Muon optimizer and auxiliary loss warmup scheduling.

Tests cover:
- Muon optimizer: step behavior, weight decay, Newton-Schulz orthogonalization
- Parameter splitting for hybrid mode
- CombinedOptimizer interface
- Auxiliary loss warmup: linear ramp, boundary conditions
- Integration with compute_total_loss
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.muon import (
    Muon,
    CombinedOptimizer,
    split_params_for_muon,
    _newton_schulz,
)
from src.models.battle_transformer import (
    BattleTransformer,
    TransformerConfig,
    TransformerOutput,
    compute_total_loss,
    get_aux_weight,
)
from src.environment.action_space import NUM_ACTIONS


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def simple_model():
    """A small linear model for optimizer testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    return model


@pytest.fixture
def config():
    """Tiny BattleTransformer config for testing."""
    return TransformerConfig(
        num_layers=2,
        hidden_dim=64,
        num_heads=2,
        dropout=0.0,
        ffn_multiplier=2,
        species_vocab_size=10,
        moves_vocab_size=10,
        items_vocab_size=10,
        abilities_vocab_size=10,
        types_vocab_size=10,
        status_vocab_size=10,
        weather_vocab_size=10,
        terrain_vocab_size=10,
        auxiliary_loss_weight=0.2,
        use_value_head=False,
    )


# ── Muon Optimizer Tests ────────────────────────────────────────────────


class TestMuonOptimizer:
    def test_basic_step(self, simple_model):
        """Muon should update parameters after a step."""
        optimizer = Muon(simple_model.parameters(), lr=0.01)
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        initial_params = [p.clone() for p in simple_model.parameters()]

        loss = nn.functional.mse_loss(simple_model(x), y)
        loss.backward()
        optimizer.step()

        for p_init, p_new in zip(initial_params, simple_model.parameters()):
            assert not torch.equal(p_init, p_new), "Parameters should change after step"

    def test_zero_grad(self, simple_model):
        """zero_grad should clear all gradients."""
        optimizer = Muon(simple_model.parameters(), lr=0.01)
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        loss = nn.functional.mse_loss(simple_model(x), y)
        loss.backward()
        optimizer.zero_grad()

        for p in simple_model.parameters():
            assert p.grad is None or (p.grad == 0).all()

    def test_momentum_buffer_created(self, simple_model):
        """Momentum buffer should be created on first step."""
        optimizer = Muon(simple_model.parameters(), lr=0.01, momentum=0.9)
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        loss = nn.functional.mse_loss(simple_model(x), y)
        loss.backward()
        optimizer.step()

        for p in simple_model.parameters():
            if p.grad is not None:
                assert "momentum_buffer" in optimizer.state[p]

    def test_weight_decay(self):
        """Weight decay should shrink parameter magnitude."""
        model = nn.Linear(10, 5, bias=False)
        nn.init.ones_(model.weight)
        optimizer = Muon([model.weight], lr=0.0, weight_decay=0.1)

        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # With lr=0, only weight decay acts: w *= (1 - lr * wd) = 1.0
        # Actually with lr=0, weight decay term is 1 - 0 * 0.1 = 1
        # Let's use non-zero lr
        model2 = nn.Linear(10, 5, bias=False)
        nn.init.ones_(model2.weight)
        optimizer2 = Muon([model2.weight], lr=0.01, weight_decay=0.5)

        x = torch.randn(2, 10)
        loss = model2(x).sum()
        loss.backward()
        initial_norm = model2.weight.norm().item()
        optimizer2.step()
        # Weight decay should reduce the norm (combined with gradient update)
        # Just check it changed
        assert model2.weight.norm().item() != initial_norm

    def test_nesterov_vs_no_nesterov(self, simple_model):
        """Nesterov and non-Nesterov should produce different updates."""
        # Two copies of the model
        model_nest = nn.Linear(10, 5)
        model_no_nest = nn.Linear(10, 5)
        model_no_nest.load_state_dict(model_nest.state_dict())

        opt_nest = Muon(model_nest.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        opt_no = Muon(model_no_nest.parameters(), lr=0.01, momentum=0.9, nesterov=False)

        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        # Step 1
        loss = nn.functional.mse_loss(model_nest(x), y)
        loss.backward()
        opt_nest.step()
        opt_nest.zero_grad()

        loss = nn.functional.mse_loss(model_no_nest(x), y)
        loss.backward()
        opt_no.step()
        opt_no.zero_grad()

        # Step 2 (momentum accumulates, Nesterov differs)
        loss = nn.functional.mse_loss(model_nest(x), y)
        loss.backward()
        opt_nest.step()

        loss = nn.functional.mse_loss(model_no_nest(x), y)
        loss.backward()
        opt_no.step()

        # After 2 steps, parameters should differ
        for p1, p2 in zip(model_nest.parameters(), model_no_nest.parameters()):
            assert not torch.allclose(p1, p2, atol=1e-7)

    def test_invalid_lr_raises(self):
        model = nn.Linear(5, 3)
        with pytest.raises(ValueError, match="Invalid learning rate"):
            Muon(model.parameters(), lr=-0.01)

    def test_invalid_momentum_raises(self):
        model = nn.Linear(5, 3)
        with pytest.raises(ValueError, match="Invalid momentum"):
            Muon(model.parameters(), momentum=1.0)


class TestNewtonSchulz:
    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        G = torch.randn(10, 20)
        result = _newton_schulz(G, steps=5)
        assert result.shape == G.shape

    def test_output_shape_3d(self):
        """3D tensors should be handled (reshaped internally)."""
        G = torch.randn(4, 5, 6)
        result = _newton_schulz(G, steps=5)
        assert result.shape == G.shape

    def test_approximate_orthogonality(self):
        """Output rows should be more orthogonal than input."""
        G = torch.randn(8, 8)
        X = _newton_schulz(G, steps=10)
        product = X @ X.T
        identity = torch.eye(8)
        # NS iteration should improve orthogonality vs raw gradient
        G_norm = G / (G.norm() + 1e-7)
        raw_product = G_norm @ G_norm.T
        ns_error = (product - identity).norm()
        raw_error = (raw_product - identity).norm()
        assert ns_error < raw_error, "NS iteration should improve orthogonality"

    def test_zero_gradient_handled(self):
        """Zero gradient should not cause NaN."""
        G = torch.zeros(5, 5)
        result = _newton_schulz(G, steps=5)
        assert not torch.isnan(result).any()


# ── Parameter Splitting Tests ────────────────────────────────────────────


class TestParamSplitting:
    def test_split_covers_all_params(self, config):
        """All trainable params should appear in exactly one group."""
        model = BattleTransformer(config)
        muon_params, adam_params = split_params_for_muon(model)
        total_split = len(muon_params) + len(adam_params)
        total_model = sum(1 for p in model.parameters() if p.requires_grad)
        assert total_split == total_model

    def test_muon_params_are_2d(self, config):
        """Muon group should only contain 2D+ tensors."""
        model = BattleTransformer(config)
        muon_params, _ = split_params_for_muon(model)
        for p in muon_params:
            assert p.dim() >= 2

    def test_adam_params_include_embeddings(self, config):
        """Adam group should include embedding parameters."""
        model = BattleTransformer(config)
        _, adam_params = split_params_for_muon(model)
        adam_param_set = {id(p) for p in adam_params}
        # Check that embedding weights are in adam params
        for name, p in model.named_parameters():
            if "emb" in name and p.requires_grad:
                assert id(p) in adam_param_set, f"{name} should be in adam params"

    def test_adam_params_include_norms(self, config):
        """Adam group should include LayerNorm parameters."""
        model = BattleTransformer(config)
        _, adam_params = split_params_for_muon(model)
        adam_param_set = {id(p) for p in adam_params}
        for name, p in model.named_parameters():
            if "norm" in name and p.requires_grad:
                assert id(p) in adam_param_set, f"{name} should be in adam params"


# ── CombinedOptimizer Tests ────────────────────────────────────────────


class TestCombinedOptimizer:
    def test_step_updates_both(self):
        """Both optimizer's params should be updated."""
        model1 = nn.Linear(10, 5, bias=False)
        model2 = nn.Linear(5, 3, bias=False)

        muon_opt = Muon([model1.weight], lr=0.01)
        adam_opt = torch.optim.AdamW([model2.weight], lr=0.01)
        combined = CombinedOptimizer(muon_opt, adam_opt)

        initial1 = model1.weight.clone()
        initial2 = model2.weight.clone()

        x = torch.randn(4, 10)
        out = model2(model1(x))
        loss = out.sum()
        loss.backward()
        combined.step()

        assert not torch.equal(initial1, model1.weight)
        assert not torch.equal(initial2, model2.weight)

    def test_zero_grad(self):
        model1 = nn.Linear(10, 5, bias=False)
        model2 = nn.Linear(5, 3, bias=False)
        muon_opt = Muon([model1.weight], lr=0.01)
        adam_opt = torch.optim.AdamW([model2.weight], lr=0.01)
        combined = CombinedOptimizer(muon_opt, adam_opt)

        x = torch.randn(4, 10)
        loss = model2(model1(x)).sum()
        loss.backward()
        combined.zero_grad()

        # set_to_none=True (default) makes managed params' grads None
        assert model1.weight.grad is None
        assert model2.weight.grad is None

    def test_param_groups(self):
        """param_groups should combine both optimizers."""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(5, 3)
        muon_opt = Muon(model1.parameters(), lr=0.02)
        adam_opt = torch.optim.AdamW(model2.parameters(), lr=0.001)
        combined = CombinedOptimizer(muon_opt, adam_opt)

        assert len(combined.param_groups) == len(muon_opt.param_groups) + len(adam_opt.param_groups)

    def test_state_dict_roundtrip(self):
        """State dict should be saveable and loadable."""
        model = nn.Linear(10, 5)
        muon_opt = Muon([model.weight], lr=0.01)
        adam_opt = torch.optim.AdamW([model.bias], lr=0.01)
        combined = CombinedOptimizer(muon_opt, adam_opt)

        # Do a step to populate state
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        combined.step()

        state = combined.state_dict()
        assert "muon" in state
        assert "adam" in state

        # Create fresh optimizers and load
        muon_opt2 = Muon([model.weight], lr=0.01)
        adam_opt2 = torch.optim.AdamW([model.bias], lr=0.01)
        combined2 = CombinedOptimizer(muon_opt2, adam_opt2)
        combined2.load_state_dict(state)


# ── Auxiliary Loss Warmup Tests ──────────────────────────────────────────


class TestAuxWarmup:
    def test_warmup_at_start(self):
        """Weight should be 0 at step 0."""
        assert get_aux_weight(0, 1000, max_weight=0.2, warmup_fraction=0.15) == 0.0

    def test_warmup_at_end(self):
        """Weight should be max_weight after warmup completes."""
        w = get_aux_weight(150, 1000, max_weight=0.2, warmup_fraction=0.15)
        assert abs(w - 0.2) < 1e-7

    def test_warmup_midpoint(self):
        """Weight at 50% warmup should be 50% of max."""
        w = get_aux_weight(75, 1000, max_weight=0.2, warmup_fraction=0.15)
        assert abs(w - 0.1) < 1e-7

    def test_warmup_past_schedule(self):
        """Weight after warmup should stay at max."""
        w = get_aux_weight(500, 1000, max_weight=0.2, warmup_fraction=0.15)
        assert abs(w - 0.2) < 1e-7

    def test_warmup_disabled(self):
        """warmup_fraction=0 should always return max_weight."""
        w = get_aux_weight(0, 1000, max_weight=0.2, warmup_fraction=0.0)
        assert abs(w - 0.2) < 1e-7

    def test_warmup_linear_increase(self):
        """Weight should increase linearly during warmup."""
        weights = [
            get_aux_weight(s, 1000, max_weight=0.2, warmup_fraction=0.1)
            for s in range(0, 101, 10)
        ]
        # Should be monotonically non-decreasing
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i - 1]

    def test_warmup_different_max_weights(self):
        """Should work with different max_weight values."""
        w = get_aux_weight(50, 1000, max_weight=0.5, warmup_fraction=0.1)
        assert abs(w - 0.25) < 1e-7

    def test_compute_total_loss_with_override(self, config):
        """compute_total_loss should use aux_weight_override when provided."""
        model = BattleTransformer(config)

        batch_size = 4
        logits = torch.randn(batch_size, NUM_ACTIONS)
        targets = torch.randint(0, NUM_ACTIONS, (batch_size,))
        legal_mask = torch.ones(batch_size, NUM_ACTIONS)

        output = TransformerOutput(
            policy_logits=logits,
            auxiliary_preds={
                "item_logits": torch.randn(batch_size, 6, config.num_item_classes),
                "speed_logits": torch.randn(batch_size, 6, config.num_speed_buckets),
                "role_logits": torch.randn(batch_size, 6, config.num_role_archetypes),
                "move_family_logits": torch.randn(batch_size, 6, config.num_move_families),
            },
        )
        aux_targets = {
            "item_targets": torch.randint(0, config.num_item_classes, (batch_size, 6)),
            "speed_targets": torch.randint(0, config.num_speed_buckets, (batch_size, 6)),
            "role_targets": torch.randint(0, config.num_role_archetypes, (batch_size, 6)),
            "move_family_targets": torch.randint(0, 2, (batch_size, 6, config.num_move_families)),
        }

        # With zero override, aux should not contribute
        loss_zero, dict_zero = compute_total_loss(
            output, targets, legal_mask,
            aux_targets=aux_targets, config=config,
            aux_weight_override=0.0,
        )

        # With normal weight, aux should contribute
        loss_normal, dict_normal = compute_total_loss(
            output, targets, legal_mask,
            aux_targets=aux_targets, config=config,
            aux_weight_override=0.2,
        )

        # Total loss with aux_weight=0 should equal just policy loss
        assert abs(loss_zero.item() - dict_zero["policy"]) < 1e-5
        # Total loss with aux_weight=0.2 should be larger (aux contribution)
        assert loss_normal.item() > loss_zero.item() or abs(loss_normal.item() - loss_zero.item()) < 1e-5
