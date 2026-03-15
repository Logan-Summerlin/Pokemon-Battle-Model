"""Tests for the battle harness and bot system.

Covers:
- Bot interface compliance
- RandomBot behavior
- MaxDamageBot behavior
- ActionMask integration with bots
- GameResult and SeriesResult aggregation
- These tests run without a Showdown server (unit tests only)
"""

import pytest

from src.environment.action_space import (
    MOVE_1,
    MOVE_2,
    MOVE_3,
    MOVE_4,
    SWITCH_2,
    SWITCH_3,
    ActionMask,
    action_from_canonical_index,
)
from src.environment.battle_env import Observation
from src.environment.state import BattleState, GamePhase, OwnPokemon, MoveSlot
from src.bots.random_bot import RandomBot
from src.bots.max_damage_bot import MaxDamageBot
from src.evaluation.battle_evaluator import GameResult, SeriesResult, TournamentResult


def _make_observation(
    legal_indices: list[int] | None = None,
    moves: list[str] | None = None,
) -> Observation:
    """Create a mock observation for testing bots."""
    state = BattleState()
    state.phase = GamePhase.BATTLE
    state.turn = 1

    # Add an active Pokemon with moves
    active = OwnPokemon()
    active.species = "Salamence"
    active.active = True
    active.current_hp = 364
    active.max_hp = 364
    if moves:
        active.moves = [MoveSlot(name=m) for m in moves]
    else:
        active.moves = [
            MoveSlot(name="Earthquake"),
            MoveSlot(name="Dragon Dance"),
            MoveSlot(name="Dragon Claw"),
            MoveSlot(name="Rock Slide"),
        ]
    state.own_team = [active]
    state.own_active_index = 0

    if legal_indices is None:
        legal_indices = [MOVE_1, MOVE_2, MOVE_3, MOVE_4, SWITCH_2, SWITCH_3]

    mask = ActionMask.from_list(legal_indices)

    return Observation(
        state=state,
        legal_actions=mask,
        turn=1,
    )


class TestRandomBot:
    """Test RandomBot behavior."""

    def test_name(self) -> None:
        bot = RandomBot()
        assert bot.name == "RandomBot"

    def test_always_chooses_legal_action(self) -> None:
        bot = RandomBot(seed=42)
        obs = _make_observation(legal_indices=[MOVE_1, MOVE_3])

        for _ in range(100):
            action = bot.choose_action(obs, obs.legal_actions)
            assert obs.legal_actions.is_legal(action.canonical_index)

    def test_deterministic_with_seed(self) -> None:
        bot1 = RandomBot(seed=12345)
        bot2 = RandomBot(seed=12345)
        obs = _make_observation()

        actions1 = [bot1.choose_action(obs, obs.legal_actions) for _ in range(20)]
        actions2 = [bot2.choose_action(obs, obs.legal_actions) for _ in range(20)]

        for a1, a2 in zip(actions1, actions2):
            assert a1.canonical_index == a2.canonical_index

    def test_uses_all_legal_actions(self) -> None:
        """Over many samples, all legal actions should be chosen at least once."""
        bot = RandomBot(seed=42)
        legal = [MOVE_1, MOVE_2, MOVE_3, MOVE_4, SWITCH_2]
        obs = _make_observation(legal_indices=legal)

        chosen = set()
        for _ in range(500):
            action = bot.choose_action(obs, obs.legal_actions)
            chosen.add(action.canonical_index)

        assert chosen == set(legal)

    def test_single_legal_action(self) -> None:
        bot = RandomBot(seed=1)
        obs = _make_observation(legal_indices=[MOVE_1])

        for _ in range(10):
            action = bot.choose_action(obs, obs.legal_actions)
            assert action.canonical_index == MOVE_1

    def test_raises_on_no_legal_actions(self) -> None:
        bot = RandomBot()
        obs = _make_observation(legal_indices=[])

        with pytest.raises(RuntimeError):
            bot.choose_action(obs, obs.legal_actions)

    def test_win_tracking(self) -> None:
        bot = RandomBot()
        bot.on_battle_start()
        bot.on_battle_end(True)
        bot.on_battle_start()
        bot.on_battle_end(False)
        bot.on_battle_start()
        bot.on_battle_end(True)

        assert bot._games_played == 3
        assert bot._wins == 2
        assert bot.win_rate == pytest.approx(2 / 3)

    def test_reset(self) -> None:
        bot = RandomBot(seed=42)
        bot.on_battle_start()
        bot.on_battle_end(True)
        bot.reset()

        assert bot._games_played == 0
        assert bot._wins == 0

    def test_random_team_order(self) -> None:
        bot = RandomBot(seed=42)
        obs = _make_observation()
        order = bot.choose_team_order(obs)
        assert len(order) == 6
        assert set(order) == set("123456")


class TestMaxDamageBot:
    """Test MaxDamageBot behavior."""

    def test_name(self) -> None:
        bot = MaxDamageBot()
        assert bot.name == "MaxDamageBot"

    def test_picks_highest_power_move(self) -> None:
        bot = MaxDamageBot(seed=1)
        obs = _make_observation(
            legal_indices=[MOVE_1, MOVE_2, MOVE_3, MOVE_4],
            moves=["Earthquake", "Dragon Dance", "Dragon Claw", "Rock Slide"],
        )

        action = bot.choose_action(obs, obs.legal_actions)
        # Earthquake has highest power (100) among these
        assert action.canonical_index == MOVE_1

    def test_falls_back_to_switch(self) -> None:
        bot = MaxDamageBot(seed=1)
        obs = _make_observation(legal_indices=[SWITCH_2, SWITCH_3])

        action = bot.choose_action(obs, obs.legal_actions)
        assert action.canonical_index in (SWITCH_2, SWITCH_3)

    def test_always_legal(self) -> None:
        bot = MaxDamageBot(seed=42)
        obs = _make_observation(legal_indices=[MOVE_2, SWITCH_3])

        for _ in range(50):
            action = bot.choose_action(obs, obs.legal_actions)
            assert obs.legal_actions.is_legal(action.canonical_index)


class TestGameResult:
    """Test GameResult dataclass."""

    def test_basic_result(self) -> None:
        result = GameResult(
            game_id=1,
            winner="p1",
            p1_bot="RandomBot",
            p2_bot="MaxDamageBot",
            total_turns=30,
            duration_seconds=5.2,
        )
        assert result.winner == "p1"
        assert result.total_turns == 30
        assert result.legality_violations == 0


class TestSeriesResult:
    """Test SeriesResult aggregation."""

    def test_win_rate_calculation(self) -> None:
        series = SeriesResult(
            p1_bot="BotA",
            p2_bot="BotB",
            total_games=100,
            p1_wins=60,
            p2_wins=35,
            ties=5,
            avg_turns=25.0,
            avg_duration=3.0,
            total_legality_violations=0,
        )
        assert series.p1_win_rate == 0.6
        assert series.p2_win_rate == 0.35

    def test_summary_format(self) -> None:
        series = SeriesResult(
            p1_bot="BotA",
            p2_bot="BotB",
            total_games=10,
            p1_wins=6,
            p2_wins=3,
            ties=1,
            avg_turns=20.0,
            avg_duration=2.0,
            total_legality_violations=0,
        )
        summary = series.summary()
        assert "BotA" in summary
        assert "BotB" in summary
        assert "6-3-1" in summary

    def test_zero_games(self) -> None:
        series = SeriesResult(
            p1_bot="A", p2_bot="B",
            total_games=0, p1_wins=0, p2_wins=0, ties=0,
            avg_turns=0, avg_duration=0, total_legality_violations=0,
        )
        assert series.p1_win_rate == 0.0
        assert series.p2_win_rate == 0.0


class TestTournamentResult:
    """Test TournamentResult aggregation."""

    def test_leaderboard(self) -> None:
        tournament = TournamentResult(
            bot_names=["A", "B", "C"],
            series_results={
                ("A", "B"): SeriesResult(
                    p1_bot="A", p2_bot="B",
                    total_games=10, p1_wins=7, p2_wins=3, ties=0,
                    avg_turns=20, avg_duration=2, total_legality_violations=0,
                ),
                ("A", "C"): SeriesResult(
                    p1_bot="A", p2_bot="C",
                    total_games=10, p1_wins=5, p2_wins=5, ties=0,
                    avg_turns=20, avg_duration=2, total_legality_violations=0,
                ),
                ("B", "C"): SeriesResult(
                    p1_bot="B", p2_bot="C",
                    total_games=10, p1_wins=6, p2_wins=4, ties=0,
                    avg_turns=20, avg_duration=2, total_legality_violations=0,
                ),
            },
        )

        board = tournament.get_leaderboard()
        # A: 12W-8L, B: 9W-11L, C: 9W-11L
        assert board[0][0] == "A"  # Most wins
        assert board[0][1] == 12  # A wins
