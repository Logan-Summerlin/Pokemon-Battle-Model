"""Battle evaluator: harness for running bot-vs-bot games.

Supports:
- Single matches between two bots
- N-game series with aggregated statistics
- Round-robin tournaments between multiple bots
- Full replay logging and per-turn action logs
- Deterministic seeds where possible
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.environment.action_space import ActionMask, action_from_canonical_index
from src.environment.battle_env import BattleEnv, Observation, StepResult
from src.environment.legality import get_legal_actions
from src.environment.showdown_client import ShowdownClient, ShowdownConfig
from src.environment.state import BattleState, BattleStateTracker, GamePhase
from src.environment.protocol import BattleMessage, MessageType
from src.bots.base_bot import Bot

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single game."""

    game_id: int
    winner: str  # "p1", "p2", or "tie"
    p1_bot: str
    p2_bot: str
    total_turns: int
    duration_seconds: float
    p1_actions: list[dict[str, Any]] = field(default_factory=list)
    p2_actions: list[dict[str, Any]] = field(default_factory=list)
    legality_violations: int = 0
    error: str = ""


@dataclass
class SeriesResult:
    """Aggregated result of a multi-game series."""

    p1_bot: str
    p2_bot: str
    total_games: int
    p1_wins: int
    p2_wins: int
    ties: int
    avg_turns: float
    avg_duration: float
    total_legality_violations: int
    game_results: list[GameResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def p1_win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.p1_wins / self.total_games

    @property
    def p2_win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.p2_wins / self.total_games

    def summary(self) -> str:
        return (
            f"{self.p1_bot} vs {self.p2_bot}: "
            f"{self.p1_wins}-{self.p2_wins}-{self.ties} "
            f"({self.total_games} games, "
            f"P1 win rate: {self.p1_win_rate:.1%}, "
            f"avg turns: {self.avg_turns:.1f}, "
            f"legality violations: {self.total_legality_violations})"
        )


@dataclass
class TournamentResult:
    """Result of a round-robin tournament."""

    bot_names: list[str]
    series_results: dict[tuple[str, str], SeriesResult] = field(default_factory=dict)
    total_games: int = 0

    def get_leaderboard(self) -> list[tuple[str, int, int, float]]:
        """Get (bot_name, wins, losses, win_rate) sorted by win rate."""
        stats: dict[str, dict[str, int]] = {}
        for name in self.bot_names:
            stats[name] = {"wins": 0, "losses": 0, "ties": 0, "games": 0}

        for (p1, p2), result in self.series_results.items():
            stats[p1]["wins"] += result.p1_wins
            stats[p1]["losses"] += result.p2_wins
            stats[p1]["ties"] += result.ties
            stats[p1]["games"] += result.total_games
            stats[p2]["wins"] += result.p2_wins
            stats[p2]["losses"] += result.p1_wins
            stats[p2]["ties"] += result.ties
            stats[p2]["games"] += result.total_games

        leaderboard = []
        for name in self.bot_names:
            s = stats[name]
            wr = s["wins"] / s["games"] if s["games"] > 0 else 0.0
            leaderboard.append((name, s["wins"], s["losses"], wr))

        leaderboard.sort(key=lambda x: x[3], reverse=True)
        return leaderboard

    def summary(self) -> str:
        lines = ["=== Tournament Results ==="]
        for rank, (name, wins, losses, wr) in enumerate(self.get_leaderboard(), 1):
            lines.append(f"  {rank}. {name}: {wins}W-{losses}L ({wr:.1%})")
        lines.append("")
        for (p1, p2), result in self.series_results.items():
            lines.append(f"  {result.summary()}")
        return "\n".join(lines)


class BattleEvaluator:
    """Runs bot-vs-bot battles on a local Showdown server.

    Usage:
        evaluator = BattleEvaluator(config)
        result = await evaluator.run_series(bot1, bot2, num_games=100)
        print(result.summary())
    """

    def __init__(
        self,
        config: ShowdownConfig | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.config = config or ShowdownConfig()
        self.log_dir = log_dir

    async def run_single_game(
        self,
        p1_bot: Bot,
        p2_bot: Bot,
        game_id: int = 0,
        p1_team: str = "",
        p2_team: str = "",
    ) -> GameResult:
        """Run a single game between two bots.

        Connects both bots to the server, manages the battle lifecycle,
        and returns the result.
        """
        start_time = time.monotonic()

        # Create two clients (one per player)
        p1_client = ShowdownClient(self.config)
        p2_client = ShowdownClient(self.config)
        p1_room = ""
        p2_room = ""

        try:
            await p1_client.connect()
            await p2_client.connect()

            p1_name = f"{p1_bot.name}_p1_{game_id}"
            p2_name = f"{p2_bot.name}_p2_{game_id}"

            await p1_client.login(p1_name)
            await p2_client.login(p2_name)

            # P1 challenges P2
            if p1_team:
                await p1_client._send_global(f"/utm {p1_team}")
            if p2_team:
                await p2_client._send_global(f"/utm {p2_team}")

            # Start the battle: send challenge, then accept, then wait for rooms.
            # Separating send from wait avoids the p1 wait consuming messages
            # before p2 has accepted.
            await p1_client._send_global(
                f"/challenge {p2_name}, {self.config.format}"
            )
            await asyncio.sleep(0.5)
            await p2_client._send_global(f"/accept {p1_name}")

            # Now wait for both battle rooms to appear
            p1_room_task = asyncio.create_task(
                p1_client._wait_for_battle_room(timeout=30)
            )
            p2_room_task = asyncio.create_task(
                p2_client._wait_for_battle_room(timeout=30)
            )
            p1_room, p2_room = await asyncio.gather(
                p1_room_task, p2_room_task
            )

            # Create environments
            p1_env = BattleEnv(p1_client, p1_room, player_id="p1")
            p2_env = BattleEnv(p2_client, p2_room, player_id="p2")

            # Notify bots
            p1_bot.on_battle_start()
            p2_bot.on_battle_start()

            # Play the battle (with per-game timeout)
            result = await asyncio.wait_for(
                self._play_battle(p1_env, p2_env, p1_bot, p2_bot, game_id),
                timeout=self.config.battle_timeout,
            )

            # Notify bots of result
            p1_won = result.winner == "p1"
            p2_won = result.winner == "p2"
            p1_bot.on_battle_end(p1_won if result.winner != "tie" else None)
            p2_bot.on_battle_end(p2_won if result.winner != "tie" else None)

            result.duration_seconds = time.monotonic() - start_time
            return result

        except Exception as e:
            logger.error("Error in game %d: %s", game_id, e)
            return GameResult(
                game_id=game_id,
                winner="",
                p1_bot=p1_bot.name,
                p2_bot=p2_bot.name,
                total_turns=0,
                duration_seconds=time.monotonic() - start_time,
                error=str(e),
            )
        finally:
            # Clean up: forfeit and leave rooms before disconnecting
            # to prevent stale room data on subsequent connections
            try:
                if p1_client.is_connected and p1_room:
                    await p1_client.forfeit(p1_room)
                    await p1_client.leave_room(p1_room)
            except Exception:
                pass
            try:
                if p2_client.is_connected and p2_room:
                    await p2_client.forfeit(p2_room)
                    await p2_client.leave_room(p2_room)
            except Exception:
                pass
            await p1_client.disconnect()
            await p2_client.disconnect()

    async def _play_battle(
        self,
        p1_env: BattleEnv,
        p2_env: BattleEnv,
        p1_bot: Bot,
        p2_bot: Bot,
        game_id: int,
    ) -> GameResult:
        """Play a battle to completion."""
        legality_violations = 0

        # Get initial observations
        p1_obs = await p1_env.reset()
        p2_obs = await p2_env.reset()

        logger.debug(
            "Game %d reset: P1 legal=%s, P2 legal=%s, P1 done=%s, P2 done=%s",
            game_id, p1_obs.legal_actions.legal_indices[:3],
            p2_obs.legal_actions.legal_indices[:3],
            p1_env.is_done, p2_env.is_done,
        )

        # Handle team preview
        if p1_obs.is_team_preview:
            p1_order = p1_bot.choose_team_order(p1_obs)
            p1_obs = await p1_env.step_team_preview(p1_order)

        if p2_obs.is_team_preview:
            p2_order = p2_bot.choose_team_order(p2_obs)
            p2_obs = await p2_env.step_team_preview(p2_order)

        # If both players have no legal actions right after reset, try
        # receiving one more update — the request may arrive in a
        # subsequent WebSocket frame.
        if not p1_obs.legal_actions.any_legal and not p2_obs.legal_actions.any_legal:
            if not p1_env.is_done and not p2_env.is_done:
                try:
                    p1_r, p2_r = await asyncio.wait_for(
                        asyncio.gather(
                            p1_env.wait_for_update(),
                            p2_env.wait_for_update(),
                        ),
                        timeout=10,
                    )
                    p1_obs = p1_r.observation
                    p2_obs = p2_r.observation
                except asyncio.TimeoutError:
                    logger.warning(
                        "Game %d: no legal actions after reset retry", game_id,
                    )

        # Main battle loop
        max_turns = 500  # Safety limit
        turn = 0

        while not p1_env.is_done and not p2_env.is_done and turn < max_turns:
            turn += 1

            p1_needs_action = p1_obs.legal_actions.any_legal
            p2_needs_action = p2_obs.legal_actions.any_legal

            # If neither player has legal actions, the game has likely ended
            # but the win message was not processed yet, or both are in a
            # transient wait state.
            if not p1_needs_action and not p2_needs_action:
                # Double-check if the game is actually done
                if p1_env.is_done or p2_env.is_done:
                    break
                # Try to receive one more update in case we missed the end
                try:
                    p1_r, p2_r = await asyncio.wait_for(
                        asyncio.gather(
                            p1_env.wait_for_update(),
                            p2_env.wait_for_update(),
                        ),
                        timeout=10,
                    )
                    p1_obs = p1_r.observation
                    p2_obs = p2_r.observation
                    continue
                except asyncio.TimeoutError:
                    logger.warning(
                        "Neither player has legal actions in game %d turn %d",
                        game_id, turn,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "Error waiting for update in game %d turn %d: %s",
                        game_id, turn, e,
                    )
                    break

            # Build the coroutine list.  Players with legal actions send a
            # choice via step(); players in a "wait" state passively receive
            # the next update via wait_for_update().
            if p1_needs_action:
                p1_action = p1_bot.choose_action(p1_obs, p1_obs.legal_actions)
                if not p1_obs.legal_actions.is_legal(p1_action.canonical_index):
                    legality_violations += 1
                    logger.warning("P1 legality violation in game %d turn %d", game_id, turn)
                p1_coro = p1_env.step(p1_action)
            else:
                p1_coro = p1_env.wait_for_update()

            if p2_needs_action:
                p2_action = p2_bot.choose_action(p2_obs, p2_obs.legal_actions)
                if not p2_obs.legal_actions.is_legal(p2_action.canonical_index):
                    legality_violations += 1
                    logger.warning("P2 legality violation in game %d turn %d", game_id, turn)
                p2_coro = p2_env.step(p2_action)
            else:
                p2_coro = p2_env.wait_for_update()

            # Execute both concurrently with a per-turn timeout
            try:
                p1_result, p2_result = await asyncio.wait_for(
                    asyncio.gather(p1_coro, p2_coro),
                    timeout=60,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Turn timeout in game %d turn %d", game_id, turn,
                )
                break

            p1_obs = p1_result.observation
            p2_obs = p2_result.observation

        # Determine winner
        winner = ""
        if p1_env.state.is_finished:
            if p1_env.state.did_we_win is True:
                winner = "p1"
            elif p1_env.state.did_we_win is False:
                winner = "p2"
            else:
                winner = "tie"
        elif p2_env.state.is_finished:
            if p2_env.state.did_we_win is True:
                winner = "p2"
            elif p2_env.state.did_we_win is False:
                winner = "p1"
            else:
                winner = "tie"

        return GameResult(
            game_id=game_id,
            winner=winner,
            p1_bot=p1_bot.name,
            p2_bot=p2_bot.name,
            total_turns=max(p1_env.state.turn, p2_env.state.turn),
            duration_seconds=0.0,  # Set by caller
            p1_actions=p1_env.get_action_log(),
            p2_actions=p2_env.get_action_log(),
            legality_violations=legality_violations,
        )

    async def run_series(
        self,
        p1_bot: Bot,
        p2_bot: Bot,
        num_games: int = 100,
        p1_team: str = "",
        p2_team: str = "",
    ) -> SeriesResult:
        """Run an N-game series between two bots.

        Games alternate sides: p1_bot plays as P1 in even games, P2 in odd.
        """
        results: list[GameResult] = []
        errors: list[str] = []

        for i in range(num_games):
            # Alternate sides for fairness
            if i % 2 == 0:
                result = await self.run_single_game(
                    p1_bot, p2_bot, game_id=i, p1_team=p1_team, p2_team=p2_team
                )
            else:
                result = await self.run_single_game(
                    p2_bot, p1_bot, game_id=i, p1_team=p2_team, p2_team=p1_team
                )
                # Flip winner since sides are swapped
                if result.winner == "p1":
                    result.winner = "p2"
                elif result.winner == "p2":
                    result.winner = "p1"

            results.append(result)
            if result.error:
                errors.append(f"Game {i}: {result.error}")

            # Brief pause between games to let the server clean up
            await asyncio.sleep(0.1)

            if (i + 1) % 10 == 0:
                logger.info(
                    "Series progress: %d/%d games complete", i + 1, num_games
                )

        # Aggregate stats
        completed = [r for r in results if not r.error]
        p1_wins = sum(1 for r in completed if r.winner == "p1")
        p2_wins = sum(1 for r in completed if r.winner == "p2")
        ties = sum(1 for r in completed if r.winner == "tie")
        avg_turns = (
            sum(r.total_turns for r in completed) / len(completed)
            if completed
            else 0.0
        )
        avg_duration = (
            sum(r.duration_seconds for r in completed) / len(completed)
            if completed
            else 0.0
        )
        total_violations = sum(r.legality_violations for r in results)

        series = SeriesResult(
            p1_bot=p1_bot.name,
            p2_bot=p2_bot.name,
            total_games=len(completed),
            p1_wins=p1_wins,
            p2_wins=p2_wins,
            ties=ties,
            avg_turns=avg_turns,
            avg_duration=avg_duration,
            total_legality_violations=total_violations,
            game_results=results,
            errors=errors,
        )

        # Log to file if log_dir is set
        if self.log_dir:
            self._save_series_log(series)

        return series

    async def run_tournament(
        self,
        bots: list[Bot],
        games_per_matchup: int = 50,
        teams: dict[str, str] | None = None,
    ) -> TournamentResult:
        """Run a round-robin tournament between all bot pairs."""
        teams = teams or {}
        tournament = TournamentResult(bot_names=[b.name for b in bots])

        for i, bot_a in enumerate(bots):
            for j, bot_b in enumerate(bots):
                if i >= j:
                    continue  # Skip self-play and duplicate matchups

                logger.info(
                    "Tournament matchup: %s vs %s (%d games)",
                    bot_a.name,
                    bot_b.name,
                    games_per_matchup,
                )

                result = await self.run_series(
                    bot_a,
                    bot_b,
                    num_games=games_per_matchup,
                    p1_team=teams.get(bot_a.name, ""),
                    p2_team=teams.get(bot_b.name, ""),
                )

                tournament.series_results[(bot_a.name, bot_b.name)] = result
                tournament.total_games += result.total_games

        return tournament

    def _save_series_log(self, series: SeriesResult) -> None:
        """Save series results to a JSON log file."""
        if not self.log_dir:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{series.p1_bot}_vs_{series.p2_bot}_{int(time.time())}.json"
        filepath = self.log_dir / filename

        log_data = {
            "p1_bot": series.p1_bot,
            "p2_bot": series.p2_bot,
            "total_games": series.total_games,
            "p1_wins": series.p1_wins,
            "p2_wins": series.p2_wins,
            "ties": series.ties,
            "p1_win_rate": series.p1_win_rate,
            "avg_turns": series.avg_turns,
            "avg_duration": series.avg_duration,
            "total_legality_violations": series.total_legality_violations,
            "games": [
                {
                    "game_id": r.game_id,
                    "winner": r.winner,
                    "total_turns": r.total_turns,
                    "duration": r.duration_seconds,
                    "legality_violations": r.legality_violations,
                    "error": r.error,
                }
                for r in series.game_results
            ],
        }

        filepath.write_text(json.dumps(log_data, indent=2))
        logger.info("Series log saved to %s", filepath)
