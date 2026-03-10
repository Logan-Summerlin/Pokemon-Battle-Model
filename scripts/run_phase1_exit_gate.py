#!/usr/bin/env python3
"""Phase 1 Exit Gate: Run 100 legal games between two scripted bots.

Requirements from IMPLEMENTATION_PLAN.md Phase 1 Exit Gate:
  "Two scripted bots can play 100 legal games to completion with zero
   crashes or legality violations. Full replay logs are generated and parseable."

This script:
1. Starts a local Pokemon Showdown server
2. Connects RandomBot vs MaxDamageBot
3. Plays 100 games (alternating sides)
4. Validates zero crashes, zero legality violations
5. Saves replay logs as JSON
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bots.max_damage_bot import MaxDamageBot
from src.bots.random_bot import RandomBot
from src.environment.showdown_client import ShowdownConfig
from src.evaluation.battle_evaluator import BattleEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase1_exit_gate")

# Suppress noisy loggers
logging.getLogger("websockets").setLevel(logging.WARNING)

SHOWDOWN_DIR = Path(os.environ.get(
    "SHOWDOWN_DIR",
    Path(__file__).resolve().parent.parent.parent / "pokemon-showdown",
))
NUM_GAMES = 100
LOG_DIR = PROJECT_ROOT / "data" / "phase1_logs"

# Use Gen 9 Random Battle format — avoids team-building/validation issues
# and provides diverse matchups automatically.
BATTLE_FORMAT = "gen9randombattle"


def start_showdown_server() -> subprocess.Popen | None:
    """Start the Pokemon Showdown server as a background process."""
    if not SHOWDOWN_DIR.exists():
        logger.error("Showdown not found at %s. Run scripts/setup_showdown.sh first.", SHOWDOWN_DIR)
        return None

    logger.info("Starting Showdown server at %s...", SHOWDOWN_DIR)
    proc = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        cwd=str(SHOWDOWN_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    # Wait for server to be ready
    time.sleep(4)
    if proc.poll() is not None:
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        logger.error("Server failed to start. stdout: %s, stderr: %s", stdout, stderr)
        return None

    logger.info("Showdown server started (PID: %d)", proc.pid)
    return proc


def stop_showdown_server(proc: subprocess.Popen) -> None:
    """Stop the Showdown server."""
    if proc and proc.poll() is None:
        logger.info("Stopping Showdown server (PID: %d)...", proc.pid)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        logger.info("Server stopped.")


async def run_exit_gate() -> dict:
    """Run the Phase 1 exit gate: 100 games between two bots."""
    config = ShowdownConfig(
        host="localhost",
        port=8000,
        format=BATTLE_FORMAT,
        connect_timeout=10.0,
        message_timeout=60.0,
        battle_timeout=600.0,
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    evaluator = BattleEvaluator(config=config, log_dir=LOG_DIR)
    random_bot = RandomBot(seed=42)
    max_dmg_bot = MaxDamageBot(seed=42)

    logger.info("=" * 60)
    logger.info("PHASE 1 EXIT GATE: %d games, RandomBot vs MaxDamageBot", NUM_GAMES)
    logger.info("=" * 60)

    start_time = time.monotonic()

    series_result = await evaluator.run_series(
        p1_bot=random_bot,
        p2_bot=max_dmg_bot,
        num_games=NUM_GAMES,
    )

    elapsed = time.monotonic() - start_time

    # Compile results
    results = {
        "exit_gate": "Phase 1",
        "total_games": series_result.total_games,
        "games_requested": NUM_GAMES,
        "p1_bot": series_result.p1_bot,
        "p2_bot": series_result.p2_bot,
        "p1_wins": series_result.p1_wins,
        "p2_wins": series_result.p2_wins,
        "ties": series_result.ties,
        "p1_win_rate": series_result.p1_win_rate,
        "avg_turns": series_result.avg_turns,
        "avg_duration_seconds": series_result.avg_duration,
        "total_legality_violations": series_result.total_legality_violations,
        "errors": series_result.errors,
        "elapsed_seconds": elapsed,
    }

    # Log results
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXIT GATE RESULTS")
    logger.info("=" * 60)
    logger.info(series_result.summary())
    logger.info("")
    logger.info("Total time: %.1f seconds (%.2f sec/game)", elapsed, elapsed / max(1, NUM_GAMES))
    logger.info("Errors: %d", len(series_result.errors))
    for err in series_result.errors:
        logger.error("  %s", err)
    logger.info("Legality violations: %d", series_result.total_legality_violations)

    # Save full results
    results_file = LOG_DIR / "phase1_exit_gate_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", results_file)

    # Check pass/fail
    passed = True
    reasons = []

    if series_result.total_games < NUM_GAMES:
        passed = False
        reasons.append(
            f"Only {series_result.total_games}/{NUM_GAMES} games completed"
        )

    if series_result.errors:
        passed = False
        reasons.append(f"{len(series_result.errors)} games had errors")

    if series_result.total_legality_violations > 0:
        passed = False
        reasons.append(
            f"{series_result.total_legality_violations} legality violations"
        )

    logger.info("")
    if passed:
        logger.info("*** PHASE 1 EXIT GATE: PASSED ***")
    else:
        logger.error("*** PHASE 1 EXIT GATE: FAILED ***")
        for reason in reasons:
            logger.error("  - %s", reason)

    results["passed"] = passed
    results["failure_reasons"] = reasons

    # Update results file with pass/fail
    results_file.write_text(json.dumps(results, indent=2))

    return results


def main() -> int:
    server_proc = None
    try:
        server_proc = start_showdown_server()
        if server_proc is None:
            logger.error("Failed to start Showdown server. Aborting.")
            return 1

        results = asyncio.run(run_exit_gate())
        return 0 if results.get("passed", False) else 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1
    finally:
        if server_proc:
            stop_showdown_server(server_proc)


if __name__ == "__main__":
    sys.exit(main())
