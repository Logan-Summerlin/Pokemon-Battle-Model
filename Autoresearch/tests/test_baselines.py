"""Tests for Phase 3 baseline components.

Tests cover:
- HeuristicBot: type effectiveness, damage estimation, switching logic
- BaselineMLP: forward pass shapes, legal masking
- BaselineGRU: forward pass with sequences
- Masked cross-entropy loss
- ModelBot: inference from live observations
- Training pipeline: smoke test on synthetic data
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.bots.heuristic_bot import (
    HeuristicBot,
    _estimate_damage,
    _type_effectiveness,
    _get_types,
    _defensive_type_score,
)
from src.bots.random_bot import RandomBot
from src.data.observation import MAX_TEAM_SIZE
from src.data.tensorizer import (
    BattleVocabularies,
    CONTEXT_FEATURE_DIM,
    FIELD_FEATURE_DIM,
    POKEMON_FEATURE_DIM,
)
from src.environment.action_space import (
    MOVE_1,
    MOVE_2,
    MOVE_3,
    MOVE_4,
    NUM_ACTIONS,
    SWITCH_2,
    SWITCH_3,
    SWITCH_4,
    ActionMask,
    BattleAction,
    ActionType,
    action_from_canonical_index,
)
from src.environment.battle_env import Observation
from src.environment.state import (
    BattleState,
    MoveSlot,
    OwnPokemon,
    OpponentPokemon,
    GamePhase,
    SideConditions,
    FieldState,
)
from src.models.baseline_mlp import (
    BaselineMLP,
    BaselineGRU,
    FLAT_INPUT_DIM,
    masked_cross_entropy,
    create_baseline_model,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_own_pokemon(
    species: str = "Salamence",
    hp: int = 300,
    max_hp: int = 300,
    moves: list[str] | None = None,
    stats: dict[str, int] | None = None,
    active: bool = True,
) -> OwnPokemon:
    """Create an OwnPokemon for testing."""
    if moves is None:
        moves = ["Earthquake", "Dragon Claw", "Dragon Dance", "Rock Slide"]
    if stats is None:
        stats = {"atk": 135, "def": 80, "spa": 110, "spd": 80, "spe": 100}

    return OwnPokemon(
        species=species,
        level=100,
        current_hp=hp,
        max_hp=max_hp,
        moves=[MoveSlot(name=m) for m in moves],
        stats=stats,
        active=active,
        fainted=hp <= 0,
    )


def _make_opp_pokemon(
    species: str = "Blissey",
    hp_frac: float = 1.0,
    active: bool = True,
) -> OpponentPokemon:
    """Create an OpponentPokemon for testing."""
    return OpponentPokemon(
        species=species,
        level=100,
        hp_fraction=hp_frac,
        active=active,
        fainted=hp_frac <= 0,
    )


def _make_state(
    own_pokemon: list[OwnPokemon] | None = None,
    opp_pokemon: list[OpponentPokemon] | None = None,
) -> BattleState:
    """Create a BattleState for testing."""
    if own_pokemon is None:
        own_pokemon = [
            _make_own_pokemon("Salamence", active=True),
            _make_own_pokemon("Skarmory", hp=300, max_hp=300, active=False,
                             moves=["Drill Peck", "Spikes", "Whirlwind", "Rest"],
                             stats={"atk": 80, "def": 140, "spa": 40, "spd": 70, "spe": 70}),
            _make_own_pokemon("Metagross", hp=300, max_hp=300, active=False,
                             moves=["Meteor Mash", "Psychic", "Earthquake", "Explosion"],
                             stats={"atk": 135, "def": 130, "spa": 95, "spd": 90, "spe": 70}),
        ]
    if opp_pokemon is None:
        opp_pokemon = [_make_opp_pokemon("Blissey")]

    state = BattleState(
        player_id="p1",
        opponent_id="p2",
        phase=GamePhase.BATTLE,
        turn=5,
        own_team=own_pokemon,
        opponent_team=opp_pokemon,
        own_active_index=0,
        opponent_active_index=0,
    )
    return state


def _make_observation(
    state: BattleState | None = None,
    legal_indices: list[int] | None = None,
) -> tuple[Observation, ActionMask]:
    """Create an Observation and ActionMask for testing."""
    if state is None:
        state = _make_state()
    if legal_indices is None:
        legal_indices = [MOVE_1, MOVE_2, MOVE_3, MOVE_4, SWITCH_2, SWITCH_3]

    mask = ActionMask.from_list(legal_indices)
    obs = Observation(state=state, legal_actions=mask, turn=state.turn)
    return obs, mask


# ── Type effectiveness tests ──────────────────────────────────────────


class TestTypeEffectiveness:
    def test_super_effective(self):
        # Fire > Grass
        assert _type_effectiveness("Fire", ["Grass"]) == 2.0

    def test_not_very_effective(self):
        # Fire > Water
        assert _type_effectiveness("Fire", ["Water"]) == 0.5

    def test_immune(self):
        # Normal > Ghost
        assert _type_effectiveness("Normal", ["Ghost"]) == 0.0

    def test_neutral(self):
        # Fire > Normal
        assert _type_effectiveness("Fire", ["Normal"]) == 1.0

    def test_dual_type(self):
        # Ground > Fire/Steel (4x)
        assert _type_effectiveness("Ground", ["Fire", "Steel"]) == 4.0

    def test_dual_type_resist(self):
        # Fire > Water/Dragon
        eff = _type_effectiveness("Fire", ["Water", "Dragon"])
        assert eff == 0.25

    def test_empty_types(self):
        assert _type_effectiveness("Fire", []) == 1.0


class TestGetTypes:
    def test_single_type(self):
        assert _get_types("Fire") == ["Fire"]

    def test_dual_type_comma(self):
        assert _get_types("Fire,Steel") == ["Fire", "Steel"]

    def test_dual_type_slash(self):
        assert _get_types("Fire/Steel") == ["Fire", "Steel"]

    def test_empty(self):
        assert _get_types("") == []


class TestDefensiveTypeScore:
    def test_neutral_matchup(self):
        score = _defensive_type_score(["Normal"], ["Normal"])
        assert score == 1.0

    def test_super_effective(self):
        score = _defensive_type_score(["Fire"], ["Water"])
        assert score == 2.0

    def test_resisted(self):
        score = _defensive_type_score(["Water"], ["Fire"])
        assert score == 1.0  # We return max(1.0, eff), so resists don't go below 1


# ── Damage estimation tests ──────────────────────────────────────────


class TestDamageEstimation:
    def test_stab_bonus(self):
        attacker = _make_own_pokemon("Salamence", stats={"atk": 135, "spa": 110})
        # Dragon Claw (Dragon) on Salamence (Dragon/Flying) = STAB
        damage = _estimate_damage("Dragon Claw", attacker, ["Normal"], ["Dragon", "Flying"])
        # Without STAB
        damage_no_stab = _estimate_damage("Rock Slide", attacker, ["Normal"], ["Dragon", "Flying"])
        # Dragon Claw has 80bp * 1.5 STAB * 1.35 atk factor
        # Rock Slide has 75bp * 1.0 * 1.35
        assert damage > damage_no_stab

    def test_type_effectiveness_in_damage(self):
        attacker = _make_own_pokemon("Salamence")
        # Earthquake on Fire type (SE)
        damage_se = _estimate_damage("Earthquake", attacker, ["Fire"], ["Dragon", "Flying"])
        # Earthquake on Flying type (immune)
        damage_immune = _estimate_damage("Earthquake", attacker, ["Flying"], ["Dragon", "Flying"])
        assert damage_se > 0
        assert damage_immune == 0

    def test_status_move_zero_damage(self):
        attacker = _make_own_pokemon("Salamence")
        damage = _estimate_damage("Swords Dance", attacker, ["Normal"], ["Dragon", "Flying"])
        assert damage == 0.0

    def test_unknown_move(self):
        attacker = _make_own_pokemon("Salamence")
        damage = _estimate_damage("UnknownMove123", attacker, ["Normal"], ["Dragon", "Flying"])
        assert damage == 60.0  # Default for unknown moves


# ── HeuristicBot tests ────────────────────────────────────────────────


class TestHeuristicBot:
    def test_picks_legal_action(self):
        bot = HeuristicBot(seed=42)
        obs, mask = _make_observation()
        action = bot.choose_action(obs, mask)
        assert mask.is_legal(action.canonical_index)

    def test_prefers_high_damage_move(self):
        """Bot should prefer Earthquake (100bp, SE vs Steel) over weak moves."""
        own = [
            _make_own_pokemon(
                "Salamence",
                moves=["Earthquake", "Dragon Claw", "Dragon Dance", "Rock Slide"],
                stats={"atk": 135, "def": 80, "spa": 110, "spd": 80, "spe": 100},
            ),
        ]
        opp = [_make_opp_pokemon("Metagross")]  # Steel/Psychic - weak to Ground
        state = _make_state(own_pokemon=own, opp_pokemon=opp)
        obs, mask = _make_observation(state, [MOVE_1, MOVE_2, MOVE_3, MOVE_4])

        bot = HeuristicBot(seed=42)
        action = bot.choose_action(obs, mask)
        # Should pick Earthquake (move 1, index 0) due to 2x SE vs Steel/Psychic
        assert action.canonical_index == MOVE_1

    def test_handles_only_switches(self):
        """Bot should handle forced switch situations."""
        obs, mask = _make_observation(legal_indices=[SWITCH_2, SWITCH_3])
        bot = HeuristicBot(seed=42)
        action = bot.choose_action(obs, mask)
        assert action.canonical_index in [SWITCH_2, SWITCH_3]

    def test_no_crash_with_empty_moves(self):
        """Bot shouldn't crash if pokemon has no moves."""
        own = [_make_own_pokemon("Ditto", moves=[])]
        opp = [_make_opp_pokemon("Blissey")]
        state = _make_state(own_pokemon=own, opp_pokemon=opp)
        obs, mask = _make_observation(state, [SWITCH_2])
        bot = HeuristicBot(seed=42)
        action = bot.choose_action(obs, mask)
        assert mask.is_legal(action.canonical_index)

    def test_bot_lifecycle(self):
        """Test battle start/end tracking."""
        bot = HeuristicBot(seed=42)
        bot.on_battle_start()
        assert bot._games_played == 1
        bot.on_battle_end(True)
        assert bot._wins == 1
        assert bot.win_rate == 1.0

    def test_reset(self):
        """Test bot reset."""
        bot = HeuristicBot(seed=42)
        bot.on_battle_start()
        bot.on_battle_end(True)
        bot.reset()
        assert bot._games_played == 0
        assert bot._wins == 0


# ── BaselineMLP tests ─────────────────────────────────────────────────


class TestBaselineMLP:
    def test_output_shape(self):
        model = BaselineMLP(hidden_dims=[64, 32])
        batch_size = 4
        own_team = torch.randn(batch_size, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        opp_team = torch.randn(batch_size, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        field = torch.randn(batch_size, FIELD_FEATURE_DIM)
        context = torch.randn(batch_size, CONTEXT_FEATURE_DIM)

        logits = model(own_team, opp_team, field, context)
        assert logits.shape == (batch_size, NUM_ACTIONS)

    def test_legal_masking(self):
        model = BaselineMLP(hidden_dims=[64, 32])
        batch_size = 2
        own_team = torch.randn(batch_size, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        opp_team = torch.randn(batch_size, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        field = torch.randn(batch_size, FIELD_FEATURE_DIM)
        context = torch.randn(batch_size, CONTEXT_FEATURE_DIM)

        # Only move 1 and switch 2 are legal
        legal_mask = torch.zeros(batch_size, NUM_ACTIONS)
        legal_mask[:, MOVE_1] = 1
        legal_mask[:, SWITCH_2] = 1

        logits = model(own_team, opp_team, field, context, legal_mask=legal_mask)

        # Illegal actions should be -inf
        for i in range(NUM_ACTIONS):
            if i not in [MOVE_1, SWITCH_2]:
                assert logits[0, i] == float("-inf")

        # Legal actions should be finite
        assert torch.isfinite(logits[0, MOVE_1])
        assert torch.isfinite(logits[0, SWITCH_2])

    def test_input_dim(self):
        expected = (
            MAX_TEAM_SIZE * POKEMON_FEATURE_DIM * 2
            + FIELD_FEATURE_DIM
            + CONTEXT_FEATURE_DIM
        )
        assert FLAT_INPUT_DIM == expected

    def test_gradient_flow(self):
        model = BaselineMLP(hidden_dims=[32])
        own = torch.randn(2, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        opp = torch.randn(2, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        field = torch.randn(2, FIELD_FEATURE_DIM)
        ctx = torch.randn(2, CONTEXT_FEATURE_DIM)
        mask = torch.ones(2, NUM_ACTIONS)
        targets = torch.tensor([0, 3])

        logits = model(own, opp, field, ctx, legal_mask=mask)
        loss = masked_cross_entropy(logits, targets, mask)
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ── BaselineGRU tests ─────────────────────────────────────────────────


class TestBaselineGRU:
    def test_output_shape(self):
        model = BaselineGRU(hidden_dim=64, num_layers=1)
        batch_size = 2
        seq_len = 5
        own = torch.randn(batch_size, seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        opp = torch.randn(batch_size, seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        field = torch.randn(batch_size, seq_len, FIELD_FEATURE_DIM)
        ctx = torch.randn(batch_size, seq_len, CONTEXT_FEATURE_DIM)

        logits = model(own, opp, field, ctx)
        assert logits.shape == (batch_size, seq_len, NUM_ACTIONS)

    def test_with_seq_len(self):
        model = BaselineGRU(hidden_dim=64, num_layers=1)
        batch_size = 2
        seq_len = 5
        own = torch.randn(batch_size, seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        opp = torch.randn(batch_size, seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        field = torch.randn(batch_size, seq_len, FIELD_FEATURE_DIM)
        ctx = torch.randn(batch_size, seq_len, CONTEXT_FEATURE_DIM)
        lengths = torch.tensor([5, 3])

        logits = model(own, opp, field, ctx, seq_len=lengths)
        assert logits.shape == (batch_size, seq_len, NUM_ACTIONS)

    def test_legal_masking_seq(self):
        model = BaselineGRU(hidden_dim=32)
        batch_size = 1
        seq_len = 3
        own = torch.randn(batch_size, seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        opp = torch.randn(batch_size, seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM)
        field = torch.randn(batch_size, seq_len, FIELD_FEATURE_DIM)
        ctx = torch.randn(batch_size, seq_len, CONTEXT_FEATURE_DIM)
        mask = torch.zeros(batch_size, seq_len, NUM_ACTIONS)
        mask[:, :, MOVE_1] = 1
        mask[:, :, MOVE_2] = 1

        logits = model(own, opp, field, ctx, legal_mask=mask)
        # Check illegal actions are -inf
        for i in range(NUM_ACTIONS):
            if i not in [MOVE_1, MOVE_2]:
                assert logits[0, 0, i] == float("-inf")


# ── Masked cross-entropy tests ───────────────────────────────────────


class TestMaskedCrossEntropy:
    def test_basic(self):
        logits = torch.randn(4, NUM_ACTIONS)
        targets = torch.tensor([0, 1, 2, 3])
        mask = torch.ones(4, NUM_ACTIONS)
        loss = masked_cross_entropy(logits, targets, mask)
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0

    def test_ignore_index(self):
        logits = torch.randn(4, NUM_ACTIONS)
        targets = torch.tensor([0, -1, 2, -1])  # -1 = padding
        mask = torch.ones(4, NUM_ACTIONS)
        loss = masked_cross_entropy(logits, targets, mask, ignore_index=-1)
        assert torch.isfinite(loss)

    def test_sequence_input(self):
        logits = torch.randn(2, 5, NUM_ACTIONS)  # batch=2, seq=5
        targets = torch.randint(0, NUM_ACTIONS, (2, 5))
        mask = torch.ones(2, 5, NUM_ACTIONS)
        loss = masked_cross_entropy(logits, targets, mask)
        assert loss.dim() == 0


# ── Factory function tests ────────────────────────────────────────────


class TestCreateModel:
    def test_create_mlp(self):
        model = create_baseline_model("mlp", hidden_dims=[64, 32])
        assert isinstance(model, BaselineMLP)

    def test_create_gru(self):
        model = create_baseline_model("gru", gru_hidden_dim=64)
        assert isinstance(model, BaselineGRU)

    def test_invalid_arch(self):
        with pytest.raises(ValueError):
            create_baseline_model("transformer")


# ── Training smoke test ───────────────────────────────────────────────


class TestTrainingSmoke:
    """Smoke test: verify training pipeline runs without crashes."""

    def _make_synthetic_data(self, n: int = 100) -> list[dict[str, np.ndarray]]:
        """Create synthetic turn data for testing."""
        rng = np.random.RandomState(42)
        turns = []
        for _ in range(n):
            legal_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
            # Make some moves and switches legal
            legal_indices = rng.choice(NUM_ACTIONS, size=rng.randint(1, 6), replace=False)
            legal_mask[legal_indices] = 1.0
            # Action is one of the legal ones
            action = np.int64(rng.choice(legal_indices))

            turns.append({
                "own_team": rng.randn(MAX_TEAM_SIZE, POKEMON_FEATURE_DIM).astype(np.float32),
                "opponent_team": rng.randn(MAX_TEAM_SIZE, POKEMON_FEATURE_DIM).astype(np.float32),
                "field": rng.randn(FIELD_FEATURE_DIM).astype(np.float32),
                "context": rng.randn(CONTEXT_FEATURE_DIM).astype(np.float32),
                "legal_mask": legal_mask,
                "action": action,
                "game_result": np.float32(rng.choice([0.0, 1.0])),
            })
        return turns

    def test_mlp_training_smoke(self):
        """MLP training should run for a few epochs without errors."""
        from src.training.bc_trainer import BCTrainer

        model = BaselineMLP(hidden_dims=[32, 16], dropout=0.0)
        train_data = self._make_synthetic_data(80)
        val_data = self._make_synthetic_data(20)

        trainer = BCTrainer(
            model=model,
            train_data=train_data,
            val_data=val_data,
            checkpoint_dir="/tmp/test_ckpt_mlp",
            learning_rate=1e-3,
            batch_size=16,
            max_epochs=3,
            early_stopping_patience=10,
            device="cpu",
        )

        result = trainer.train()
        assert result.total_epochs == 3
        assert result.best_val_loss < float("inf")
        assert result.final_train_accuracy >= 0.0

    def test_gru_training_smoke(self):
        """GRU training on sequences should run without errors."""
        from src.training.bc_trainer import BCTrainer

        model = BaselineGRU(hidden_dim=32, num_layers=1, dropout=0.0)

        # Create sequence data
        rng = np.random.RandomState(42)
        sequences = []
        for _ in range(20):
            seq_len = rng.randint(3, 10)
            legal_masks = np.zeros((seq_len, NUM_ACTIONS), dtype=np.float32)
            actions = np.full(seq_len, -1, dtype=np.int64)
            for t in range(seq_len):
                legal = rng.choice(NUM_ACTIONS, size=rng.randint(1, 5), replace=False)
                legal_masks[t, legal] = 1.0
                actions[t] = rng.choice(legal)

            sequences.append({
                "own_team": rng.randn(seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM).astype(np.float32),
                "opponent_team": rng.randn(seq_len, MAX_TEAM_SIZE, POKEMON_FEATURE_DIM).astype(np.float32),
                "field": rng.randn(seq_len, FIELD_FEATURE_DIM).astype(np.float32),
                "context": rng.randn(seq_len, CONTEXT_FEATURE_DIM).astype(np.float32),
                "legal_mask": legal_masks,
                "action": actions,
                "game_result": np.full(seq_len, rng.choice([0.0, 1.0]), dtype=np.float32),
                "seq_len": np.int64(seq_len),
            })

        trainer = BCTrainer(
            model=model,
            train_data=sequences[:16],
            val_data=sequences[16:],
            checkpoint_dir="/tmp/test_ckpt_gru",
            learning_rate=1e-3,
            batch_size=4,
            max_epochs=2,
            early_stopping_patience=10,
            sequence_mode=True,
            device="cpu",
        )

        result = trainer.train()
        assert result.total_epochs == 2

    def test_model_can_overfit_small_data(self):
        """Verify model can learn to overfit a tiny dataset (sanity check)."""
        model = BaselineMLP(hidden_dims=[64, 32], dropout=0.0)
        # Create a very simple dataset: always action 0 when legal
        data = []
        for _ in range(50):
            legal_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
            legal_mask[0] = 1.0
            legal_mask[1] = 1.0
            data.append({
                "own_team": np.random.randn(MAX_TEAM_SIZE, POKEMON_FEATURE_DIM).astype(np.float32),
                "opponent_team": np.random.randn(MAX_TEAM_SIZE, POKEMON_FEATURE_DIM).astype(np.float32),
                "field": np.random.randn(FIELD_FEATURE_DIM).astype(np.float32),
                "context": np.random.randn(CONTEXT_FEATURE_DIM).astype(np.float32),
                "legal_mask": legal_mask,
                "action": np.int64(0),  # Always action 0
                "game_result": np.float32(1.0),
            })

        from src.training.bc_trainer import BCTrainer

        trainer = BCTrainer(
            model=model,
            train_data=data,
            val_data=data,  # Same data for overfit test
            checkpoint_dir="/tmp/test_ckpt_overfit",
            learning_rate=1e-2,
            batch_size=50,
            max_epochs=50,
            early_stopping_patience=100,
            device="cpu",
        )

        result = trainer.train()
        # Should achieve high accuracy on this trivial task
        assert result.final_train_accuracy > 0.8


# ── RandomBot consistency tests ───────────────────────────────────────


class TestRandomBot:
    def test_picks_legal_action(self):
        bot = RandomBot(seed=42)
        obs, mask = _make_observation()
        for _ in range(20):
            action = bot.choose_action(obs, mask)
            assert mask.is_legal(action.canonical_index)

    def test_deterministic_with_seed(self):
        bot1 = RandomBot(seed=42)
        bot2 = RandomBot(seed=42)
        obs, mask = _make_observation()
        for _ in range(10):
            a1 = bot1.choose_action(obs, mask)
            a2 = bot2.choose_action(obs, mask)
            assert a1.canonical_index == a2.canonical_index

    def test_uniform_distribution(self):
        """Random bot should roughly uniformly sample legal actions."""
        bot = RandomBot(seed=123)
        legal_indices = [MOVE_1, MOVE_2, SWITCH_2]
        obs, mask = _make_observation(legal_indices=legal_indices)

        counts = {i: 0 for i in legal_indices}
        n = 3000
        for _ in range(n):
            action = bot.choose_action(obs, mask)
            counts[action.canonical_index] += 1

        # Each should be roughly 1/3 (allow 15% tolerance)
        for idx, count in counts.items():
            frac = count / n
            assert 0.2 < frac < 0.5, f"Action {idx}: {frac:.2f} (expected ~0.33)"
