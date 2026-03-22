"""First-person battle state tracking.

Maintains the battle state as seen from one player's perspective only.
This is the single most important engineering constraint: the state must
reconstruct only what the acting player knew at decision time.

Hidden Information Doctrine:
- Own team: full info (HP, moves, items, abilities, stats, status)
- Opponent team: only what has been revealed through battle events
- Unrevealed info uses explicit "unknown" markers, never zeros or defaults
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any

from src.environment.protocol import (
    BattleMessage,
    HPStatus,
    MessageType,
    PokemonDetails,
    PokemonIdent,
    parse_ability_message,
    parse_boost_message,
    parse_damage_message,
    parse_field_message,
    parse_item_message,
    parse_move_message,
    parse_poke_message,
    parse_side_condition_message,
    parse_status_message,
    parse_switch_message,
    parse_weather_message,
)


# ── Constants ───────────────────────────────────────────────────────────────

UNKNOWN = "unknown"
MAX_TEAM_SIZE = 6
MAX_MOVES = 4
STAT_NAMES = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


class GamePhase(Enum):
    """Phases of a battle."""

    TEAM_PREVIEW = "team_preview"
    BATTLE = "battle"
    FINISHED = "finished"


# ── Pokemon state dataclasses ───────────────────────────────────────────────


@dataclass
class MoveSlot:
    """A known move slot."""

    name: str
    pp: int = -1  # -1 means unknown remaining PP
    max_pp: int = -1
    disabled: bool = False


@dataclass
class OwnPokemon:
    """Full state for a Pokemon on our team (complete information)."""

    species: str = ""
    level: int = 100
    gender: str = ""
    current_hp: int = 0
    max_hp: int = 0
    status: str = ""  # brn, par, slp, psn, tox, frz, or ""
    item: str = ""
    ability: str = ""
    moves: list[MoveSlot] = dc_field(default_factory=list)
    boosts: dict[str, int] = dc_field(default_factory=dict)
    active: bool = False
    fainted: bool = False
    # Volatile statuses (e.g., confusion, leech seed, substitute)
    volatiles: set[str] = dc_field(default_factory=set)
    # Stats from request (base stats with nature/EVs/IVs applied)
    stats: dict[str, int] = dc_field(default_factory=dict)

    @property
    def hp_fraction(self) -> float:
        if self.max_hp == 0:
            return 0.0
        return self.current_hp / self.max_hp

    @property
    def is_alive(self) -> bool:
        return not self.fainted and self.current_hp > 0


@dataclass
class OpponentPokemon:
    """Partial state for an opponent's Pokemon (first-person only).

    Uses UNKNOWN markers for unrevealed information.
    """

    species: str = ""
    level: int = 100
    gender: str = ""
    hp_fraction: float = 1.0  # We only see HP as a percentage
    status: str = ""
    # Revealed information — starts as UNKNOWN
    item: str = UNKNOWN
    ability: str = UNKNOWN
    revealed_moves: list[str] = dc_field(default_factory=list)
    boosts: dict[str, int] = dc_field(default_factory=dict)
    active: bool = False
    fainted: bool = False
    volatiles: set[str] = dc_field(default_factory=set)
    # Track whether this Pokemon has been seen in battle (switched in at least once)
    seen_in_battle: bool = False

    @property
    def is_alive(self) -> bool:
        return not self.fainted and self.hp_fraction > 0

    def reveal_move(self, move_name: str) -> None:
        """Record a newly revealed move."""
        if move_name not in self.revealed_moves:
            self.revealed_moves.append(move_name)

    def reveal_item(self, item_name: str) -> None:
        """Record a revealed item."""
        self.item = item_name

    def reveal_ability(self, ability_name: str) -> None:
        """Record a revealed ability."""
        self.ability = ability_name


# ── Battlefield conditions ──────────────────────────────────────────────────


@dataclass
class SideConditions:
    """Conditions active on one side of the field."""

    stealth_rock: bool = False
    spikes: int = 0  # 0-3 layers
    toxic_spikes: int = 0  # 0-2 layers
    sticky_web: bool = False
    reflect: int = 0  # Remaining turns (0 = inactive)
    light_screen: int = 0
    aurora_veil: int = 0
    tailwind: int = 0
    # Other side conditions
    other: set[str] = dc_field(default_factory=set)


@dataclass
class FieldState:
    """Global battlefield conditions."""

    weather: str = ""  # "", "RainDance", "SunnyDay", "Sandstorm", "Snow"
    weather_turns: int = 0
    terrain: str = ""  # "", "Electric Terrain", "Grassy Terrain", etc.
    terrain_turns: int = 0
    trick_room: int = 0  # Remaining turns
    gravity: int = 0
    # Other field conditions
    other: set[str] = dc_field(default_factory=set)


# ── Turn history ────────────────────────────────────────────────────────────


@dataclass
class TurnAction:
    """Record of actions and events in a single turn."""

    turn_number: int
    our_action: str = ""  # "move Earthquake", "switch Pikachu", etc.
    opponent_action: str = ""  # What we observed the opponent doing
    events: list[str] = dc_field(default_factory=list)  # Key events
    speed_order: str = ""  # "us_first", "them_first", or ""


# ── Main battle state ───────────────────────────────────────────────────────


# Mapping from side condition protocol names to SideConditions fields
_SIDE_CONDITION_MAP: dict[str, str] = {
    "Stealth Rock": "stealth_rock",
    "Spikes": "spikes",
    "Toxic Spikes": "toxic_spikes",
    "Sticky Web": "sticky_web",
    "Reflect": "reflect",
    "Light Screen": "light_screen",
    "Aurora Veil": "aurora_veil",
    "Tailwind": "tailwind",
    "move: Stealth Rock": "stealth_rock",
    "move: Spikes": "spikes",
    "move: Toxic Spikes": "toxic_spikes",
    "move: Sticky Web": "sticky_web",
    "move: Reflect": "reflect",
    "move: Light Screen": "light_screen",
    "move: Aurora Veil": "aurora_veil",
    "move: Tailwind": "tailwind",
}

# Conditions that stack (layers)
_STACKABLE_CONDITIONS = {"spikes", "toxic_spikes"}

# Conditions that have turn durations
_TIMED_CONDITIONS = {"reflect", "light_screen", "aurora_veil", "tailwind"}


@dataclass
class BattleState:
    """Complete first-person battle state.

    Tracks everything from one player's perspective, maintaining the
    hidden information doctrine: own team has full info, opponent team
    has only revealed info with explicit unknown markers.
    """

    # Identity
    player_id: str = ""  # "p1" or "p2"
    opponent_id: str = ""  # "p2" or "p1"
    battle_id: str = ""

    # Game phase
    phase: GamePhase = GamePhase.TEAM_PREVIEW
    turn: int = 0
    winner: str = ""  # "", player_id, or opponent_id

    # Teams
    own_team: list[OwnPokemon] = dc_field(default_factory=list)
    opponent_team: list[OpponentPokemon] = dc_field(default_factory=list)

    # Active Pokemon indices
    own_active_index: int = -1
    opponent_active_index: int = -1

    # Field conditions
    own_side: SideConditions = dc_field(default_factory=SideConditions)
    opponent_side: SideConditions = dc_field(default_factory=SideConditions)
    field: FieldState = dc_field(default_factory=FieldState)

    # History
    turn_history: list[TurnAction] = dc_field(default_factory=list)

    # Team preview info (species seen at preview)
    preview_species: list[str] = dc_field(default_factory=list)
    opponent_preview_species: list[str] = dc_field(default_factory=list)

    # Request data from server (raw, for current decision)
    _current_request: dict[str, Any] = dc_field(default_factory=dict)

    # Player name to ID mapping (populated from |player| messages)
    _player_names: dict[str, str] = dc_field(default_factory=dict)  # name -> id

    @property
    def own_active(self) -> OwnPokemon | None:
        if 0 <= self.own_active_index < len(self.own_team):
            return self.own_team[self.own_active_index]
        return None

    @property
    def opponent_active(self) -> OpponentPokemon | None:
        if 0 <= self.opponent_active_index < len(self.opponent_team):
            return self.opponent_team[self.opponent_active_index]
        return None

    @property
    def own_alive_count(self) -> int:
        return sum(1 for p in self.own_team if p.is_alive)

    @property
    def opponent_alive_count(self) -> int:
        return sum(1 for p in self.opponent_team if p.is_alive)

    @property
    def is_finished(self) -> bool:
        return self.phase == GamePhase.FINISHED

    @property
    def did_we_win(self) -> bool | None:
        if not self.is_finished:
            return None
        return self.winner == self.player_id

    def get_side_conditions(self, player: str) -> SideConditions:
        """Get side conditions for a player."""
        if player == self.player_id:
            return self.own_side
        return self.opponent_side

    def _find_own_pokemon_by_species(self, species: str) -> int:
        """Find index of own Pokemon by species name."""
        for i, poke in enumerate(self.own_team):
            if poke.species == species:
                return i
        return -1

    def _find_own_pokemon_by_name(self, name: str) -> int:
        """Find index of own Pokemon by display name (nickname)."""
        for i, poke in enumerate(self.own_team):
            if poke.species == name or poke.species.startswith(name.split("-")[0]):
                return i
        return -1

    def _find_opponent_pokemon_by_species(self, species: str) -> int:
        """Find index of opponent Pokemon by species."""
        for i, poke in enumerate(self.opponent_team):
            if poke.species == species:
                return i
        return -1

    def _find_opponent_pokemon_by_name(self, name: str) -> int:
        """Find index of opponent Pokemon by display name."""
        for i, poke in enumerate(self.opponent_team):
            if poke.species == name or poke.species.startswith(name.split("-")[0]):
                return i
        return -1

    def snapshot(self) -> BattleState:
        """Create a deep copy of the current state."""
        return deepcopy(self)


class BattleStateTracker:
    """Tracks battle state by processing protocol messages.

    Maintains first-person state for the specified player, updating
    as messages arrive from the Showdown server.
    """

    def __init__(self, player_id: str = "p1") -> None:
        self.state = BattleState()
        self.state.player_id = player_id
        self.state.opponent_id = "p2" if player_id == "p1" else "p1"

    def _is_our_pokemon(self, ident: PokemonIdent | None) -> bool:
        """Check if a Pokemon ident belongs to our player."""
        return ident is not None and ident.player == self.state.player_id

    def _is_opponent_pokemon(self, ident: PokemonIdent | None) -> bool:
        """Check if a Pokemon ident belongs to the opponent."""
        return ident is not None and ident.player == self.state.opponent_id

    def process_message(self, msg: BattleMessage) -> None:
        """Process a single protocol message and update state."""
        handler = _MESSAGE_HANDLERS.get(msg.msg_type)
        if handler:
            handler(self, msg)

    def process_messages(self, messages: list[BattleMessage]) -> None:
        """Process a sequence of protocol messages."""
        for msg in messages:
            self.process_message(msg)

    def update_from_request(self, request_json: str) -> None:
        """Update own team state from a |request| JSON payload.

        The request contains the authoritative state of our own team,
        including full move info, stats, items, and abilities.
        """
        try:
            data = json.loads(request_json)
        except (json.JSONDecodeError, TypeError):
            return

        self.state._current_request = data

        if data.get("wait"):
            return

        # Force switch request
        if data.get("forceSwitch"):
            pass  # Legal actions handled by legality module

        # Team preview request
        if data.get("teamPreview"):
            self._update_team_from_request(data)
            self.state.phase = GamePhase.TEAM_PREVIEW
            return

        # Normal active request
        self._update_team_from_request(data)
        if self.state.phase == GamePhase.TEAM_PREVIEW:
            self.state.phase = GamePhase.BATTLE

    def _update_team_from_request(self, data: dict[str, Any]) -> None:
        """Update own team from request data."""
        side = data.get("side", {})
        pokemon_list = side.get("pokemon", [])

        if not pokemon_list:
            return

        # Set player info from request
        if side.get("id"):
            self.state.player_id = side["id"]
            self.state.opponent_id = "p2" if side["id"] == "p1" else "p1"

        # Initialize team if empty
        if not self.state.own_team:
            self.state.own_team = [OwnPokemon() for _ in range(len(pokemon_list))]

        for i, poke_data in enumerate(pokemon_list):
            if i >= len(self.state.own_team):
                self.state.own_team.append(OwnPokemon())

            poke = self.state.own_team[i]

            # Parse ident and details
            ident_str = poke_data.get("ident", "")
            details = poke_data.get("details", "")
            condition = poke_data.get("condition", "")

            # Species from details
            if details:
                parts = details.split(",")
                poke.species = parts[0].strip()
                for part in parts[1:]:
                    part = part.strip()
                    if part.startswith("L"):
                        try:
                            poke.level = int(part[1:])
                        except ValueError:
                            pass
                    elif part in ("M", "F"):
                        poke.gender = part

            # HP and status from condition
            if condition:
                if condition == "0 fnt":
                    poke.current_hp = 0
                    poke.max_hp = poke.max_hp or 1
                    poke.fainted = True
                    poke.status = ""
                else:
                    hp_parts = condition.split()
                    if "/" in hp_parts[0]:
                        hp_vals = hp_parts[0].split("/")
                        poke.current_hp = int(hp_vals[0])
                        poke.max_hp = int(hp_vals[1])
                    if len(hp_parts) > 1:
                        poke.status = hp_parts[1]
                    else:
                        poke.status = ""
                    poke.fainted = poke.current_hp == 0

            # Active status
            poke.active = poke_data.get("active", False)
            if poke.active:
                self.state.own_active_index = i

            # Stats
            stats = poke_data.get("stats", {})
            if stats:
                poke.stats = dict(stats)

            # Moves
            moves_data = poke_data.get("moves", [])
            if moves_data:
                poke.moves = []
                for move_info in moves_data:
                    if isinstance(move_info, dict):
                        slot = MoveSlot(
                            name=move_info.get("move", move_info.get("id", "")),
                            pp=move_info.get("pp", -1),
                            max_pp=move_info.get("maxpp", -1),
                            disabled=move_info.get("disabled", False),
                        )
                    else:
                        slot = MoveSlot(name=str(move_info))
                    poke.moves.append(slot)

            # Item and ability
            if "item" in poke_data:
                poke.item = poke_data["item"]
            if "ability" in poke_data:
                poke.ability = poke_data["ability"]
            if "baseAbility" in poke_data:
                if not poke.ability:
                    poke.ability = poke_data["baseAbility"]

    # ── Message handlers ────────────────────────────────────────────────

    def _handle_player(self, msg: BattleMessage) -> None:
        """Handle |player| message.

        Format: |player|p1|Username|avatar|rating
        Stores the username→player_id mapping for winner resolution.
        """
        if len(msg.args) >= 2:
            player_id = msg.args[0].strip()  # "p1" or "p2"
            username = msg.args[1].strip()
            if player_id and username:
                self.state._player_names[username] = player_id

    def _handle_poke(self, msg: BattleMessage) -> None:
        """Handle |poke| for team preview."""
        data = parse_poke_message(msg)
        player = data.get("player", "")
        details = data.get("details")
        if not details:
            return

        if player == self.state.player_id:
            self.state.preview_species.append(details.species)
        elif player == self.state.opponent_id:
            self.state.opponent_preview_species.append(details.species)
            # Initialize opponent team slot
            opp = OpponentPokemon(
                species=details.species,
                level=details.level,
                gender=details.gender,
            )
            self.state.opponent_team.append(opp)

    def _handle_teampreview(self, msg: BattleMessage) -> None:
        """Handle |teampreview| message."""
        self.state.phase = GamePhase.TEAM_PREVIEW

    def _handle_start(self, msg: BattleMessage) -> None:
        """Handle |start| message."""
        self.state.phase = GamePhase.BATTLE

    def _handle_turn(self, msg: BattleMessage) -> None:
        """Handle |turn| message."""
        if msg.args:
            try:
                self.state.turn = int(msg.args[0].strip())
            except ValueError:
                pass
        self.state.phase = GamePhase.BATTLE
        # Start tracking new turn
        self.state.turn_history.append(TurnAction(turn_number=self.state.turn))

    def _handle_switch(self, msg: BattleMessage) -> None:
        """Handle |switch| or |drag| message."""
        data = parse_switch_message(msg)
        ident = data.get("ident")
        details = data.get("details")
        hp_status = data.get("hp_status")
        if not ident or not details:
            return

        if self._is_our_pokemon(ident):
            idx = self.state._find_own_pokemon_by_species(details.species)
            if idx >= 0:
                # Mark old active as inactive
                if self.state.own_active is not None:
                    self.state.own_active.active = False
                    self.state.own_active.boosts = {}
                    self.state.own_active.volatiles = set()
                poke = self.state.own_team[idx]
                poke.active = True
                self.state.own_active_index = idx
                if hp_status:
                    poke.current_hp = hp_status.current_hp
                    poke.max_hp = hp_status.max_hp
                    poke.status = hp_status.status if hp_status.status != "fnt" else ""
                    poke.fainted = hp_status.is_fainted
        elif self._is_opponent_pokemon(ident):
            idx = self.state._find_opponent_pokemon_by_species(details.species)
            if idx < 0:
                # First time seeing this species (shouldn't happen with team preview)
                opp = OpponentPokemon(
                    species=details.species,
                    level=details.level,
                    gender=details.gender,
                )
                self.state.opponent_team.append(opp)
                idx = len(self.state.opponent_team) - 1

            # Mark old active as inactive
            if self.state.opponent_active is not None:
                self.state.opponent_active.active = False
                self.state.opponent_active.boosts = {}
                self.state.opponent_active.volatiles = set()

            opp = self.state.opponent_team[idx]
            opp.active = True
            opp.seen_in_battle = True
            self.state.opponent_active_index = idx
            if hp_status:
                if hp_status.max_hp > 0:
                    opp.hp_fraction = hp_status.current_hp / hp_status.max_hp
                opp.status = hp_status.status if hp_status.status != "fnt" else ""
                opp.fainted = hp_status.is_fainted

            # Record in turn history
            if self.state.turn_history:
                self.state.turn_history[-1].opponent_action = (
                    f"switch {details.species}"
                )

    def _handle_move(self, msg: BattleMessage) -> None:
        """Handle |move| message."""
        data = parse_move_message(msg)
        source = data.get("source")
        move = data.get("move", "")

        if self._is_opponent_pokemon(source) and self.state.opponent_active:
            self.state.opponent_active.reveal_move(move)
            if self.state.turn_history:
                self.state.turn_history[-1].opponent_action = f"move {move}"
        elif self._is_our_pokemon(source):
            if self.state.turn_history:
                self.state.turn_history[-1].our_action = f"move {move}"

    def _handle_damage(self, msg: BattleMessage) -> None:
        """Handle |-damage| message."""
        data = parse_damage_message(msg)
        target = data.get("target")
        hp_status: HPStatus | None = data.get("hp_status")
        if not target or not hp_status:
            return

        if self._is_our_pokemon(target):
            idx = self.state.own_active_index
            if idx >= 0:
                poke = self.state.own_team[idx]
                poke.current_hp = hp_status.current_hp
                poke.max_hp = hp_status.max_hp
                if hp_status.status:
                    poke.status = hp_status.status
                poke.fainted = hp_status.is_fainted
        elif self._is_opponent_pokemon(target):
            idx = self.state.opponent_active_index
            if idx >= 0:
                opp = self.state.opponent_team[idx]
                if hp_status.max_hp > 0:
                    opp.hp_fraction = hp_status.current_hp / hp_status.max_hp
                elif hp_status.is_fainted:
                    opp.hp_fraction = 0.0
                if hp_status.status:
                    opp.status = hp_status.status
                opp.fainted = hp_status.is_fainted

    def _handle_heal(self, msg: BattleMessage) -> None:
        """Handle |-heal| message (same format as damage)."""
        self._handle_damage(msg)

    def _handle_faint(self, msg: BattleMessage) -> None:
        """Handle |faint| message."""
        if not msg.args:
            return
        from src.environment.protocol import parse_pokemon_ident

        ident = parse_pokemon_ident(msg.args[0])
        if not ident:
            return

        if self._is_our_pokemon(ident):
            idx = self.state.own_active_index
            if idx >= 0:
                self.state.own_team[idx].fainted = True
                self.state.own_team[idx].current_hp = 0
        elif self._is_opponent_pokemon(ident):
            idx = self.state.opponent_active_index
            if idx >= 0:
                self.state.opponent_team[idx].fainted = True
                self.state.opponent_team[idx].hp_fraction = 0.0

    def _handle_status(self, msg: BattleMessage) -> None:
        """Handle |-status| message."""
        data = parse_status_message(msg)
        target = data.get("target")
        status = data.get("status", "")
        if not target:
            return

        if self._is_our_pokemon(target):
            if self.state.own_active:
                self.state.own_active.status = status
        elif self._is_opponent_pokemon(target):
            if self.state.opponent_active:
                self.state.opponent_active.status = status

    def _handle_curestatus(self, msg: BattleMessage) -> None:
        """Handle |-curestatus| message."""
        data = parse_status_message(msg)
        target = data.get("target")
        if not target:
            return

        if self._is_our_pokemon(target):
            # Could be bench Pokemon getting cured (e.g., Heal Bell)
            for poke in self.state.own_team:
                if poke.status == data.get("status", ""):
                    poke.status = ""
        elif self._is_opponent_pokemon(target):
            if self.state.opponent_active:
                self.state.opponent_active.status = ""

    def _handle_boost(self, msg: BattleMessage) -> None:
        """Handle |-boost| message."""
        data = parse_boost_message(msg)
        target = data.get("target")
        stat = data.get("stat", "")
        amount = data.get("amount", 0)
        if not target or not stat:
            return

        if self._is_our_pokemon(target) and self.state.own_active:
            current = self.state.own_active.boosts.get(stat, 0)
            self.state.own_active.boosts[stat] = min(6, current + amount)
        elif self._is_opponent_pokemon(target) and self.state.opponent_active:
            current = self.state.opponent_active.boosts.get(stat, 0)
            self.state.opponent_active.boosts[stat] = min(6, current + amount)

    def _handle_unboost(self, msg: BattleMessage) -> None:
        """Handle |-unboost| message."""
        data = parse_boost_message(msg)
        target = data.get("target")
        stat = data.get("stat", "")
        amount = data.get("amount", 0)
        if not target or not stat:
            return

        if self._is_our_pokemon(target) and self.state.own_active:
            current = self.state.own_active.boosts.get(stat, 0)
            self.state.own_active.boosts[stat] = max(-6, current - amount)
        elif self._is_opponent_pokemon(target) and self.state.opponent_active:
            current = self.state.opponent_active.boosts.get(stat, 0)
            self.state.opponent_active.boosts[stat] = max(-6, current - amount)

    def _handle_clearboost(self, msg: BattleMessage) -> None:
        """Handle |-clearboost| or |-clearallboost| message."""
        if not msg.args:
            return
        from src.environment.protocol import parse_pokemon_ident

        ident = parse_pokemon_ident(msg.args[0])
        if not ident:
            return
        if self._is_our_pokemon(ident) and self.state.own_active:
            self.state.own_active.boosts = {}
        elif self._is_opponent_pokemon(ident) and self.state.opponent_active:
            self.state.opponent_active.boosts = {}

    def _handle_weather(self, msg: BattleMessage) -> None:
        """Handle |-weather| message."""
        data = parse_weather_message(msg)
        weather = data.get("weather", "")
        if weather == "none":
            self.state.field.weather = ""
            self.state.field.weather_turns = 0
        elif "upkeep" not in msg.kwargs:
            self.state.field.weather = weather
            self.state.field.weather_turns = 1
        else:
            self.state.field.weather_turns += 1

    def _handle_fieldstart(self, msg: BattleMessage) -> None:
        """Handle |-fieldstart| message."""
        data = parse_field_message(msg)
        condition = data.get("condition", "")
        condition_lower = condition.lower()
        if "trick room" in condition_lower:
            self.state.field.trick_room = 5
        elif "gravity" in condition_lower:
            self.state.field.gravity = 5
        elif "terrain" in condition_lower:
            self.state.field.terrain = condition
            self.state.field.terrain_turns = 1
        else:
            self.state.field.other.add(condition)

    def _handle_fieldend(self, msg: BattleMessage) -> None:
        """Handle |-fieldend| message."""
        data = parse_field_message(msg)
        condition = data.get("condition", "")
        condition_lower = condition.lower()
        if "trick room" in condition_lower:
            self.state.field.trick_room = 0
        elif "gravity" in condition_lower:
            self.state.field.gravity = 0
        elif "terrain" in condition_lower:
            self.state.field.terrain = ""
            self.state.field.terrain_turns = 0
        else:
            self.state.field.other.discard(condition)

    def _handle_sidestart(self, msg: BattleMessage) -> None:
        """Handle |-sidestart| message."""
        data = parse_side_condition_message(msg)
        player = data.get("player", "")
        condition = data.get("condition", "")
        if not player or not condition:
            return

        side = self.state.get_side_conditions(player)
        field_name = _SIDE_CONDITION_MAP.get(condition)
        if field_name:
            if field_name in _STACKABLE_CONDITIONS:
                current = getattr(side, field_name, 0)
                max_layers = 3 if field_name == "spikes" else 2
                setattr(side, field_name, min(max_layers, current + 1))
            elif field_name in _TIMED_CONDITIONS:
                setattr(side, field_name, 5)  # Default screen/tailwind duration
            else:
                setattr(side, field_name, True)
        else:
            side.other.add(condition)

    def _handle_sideend(self, msg: BattleMessage) -> None:
        """Handle |-sideend| message."""
        data = parse_side_condition_message(msg)
        player = data.get("player", "")
        condition = data.get("condition", "")
        if not player or not condition:
            return

        side = self.state.get_side_conditions(player)
        field_name = _SIDE_CONDITION_MAP.get(condition)
        if field_name:
            if field_name in _STACKABLE_CONDITIONS:
                setattr(side, field_name, 0)
            elif field_name in _TIMED_CONDITIONS:
                setattr(side, field_name, 0)
            else:
                setattr(side, field_name, False)
        else:
            side.other.discard(condition)

    def _handle_item(self, msg: BattleMessage) -> None:
        """Handle |-item| message (item revealed)."""
        data = parse_item_message(msg)
        target = data.get("target")
        item = data.get("item", "")
        if not target or not item:
            return

        if self._is_opponent_pokemon(target) and self.state.opponent_active:
            self.state.opponent_active.reveal_item(item)

    def _handle_enditem(self, msg: BattleMessage) -> None:
        """Handle |-enditem| message (item consumed/knocked off)."""
        data = parse_item_message(msg)
        target = data.get("target")
        item = data.get("item", "")
        if not target:
            return

        if self._is_our_pokemon(target) and self.state.own_active:
            self.state.own_active.item = ""
        elif self._is_opponent_pokemon(target) and self.state.opponent_active:
            # Record that we now know the item was consumed
            if item:
                self.state.opponent_active.reveal_item(item)
            self.state.opponent_active.item = f"(consumed: {item})" if item else ""

    def _handle_ability(self, msg: BattleMessage) -> None:
        """Handle |-ability| message (ability revealed)."""
        data = parse_ability_message(msg)
        target = data.get("target")
        ability = data.get("ability", "")
        if not target or not ability:
            return

        if self._is_opponent_pokemon(target) and self.state.opponent_active:
            self.state.opponent_active.reveal_ability(ability)

    def _handle_start_effect(self, msg: BattleMessage) -> None:
        """Handle |-start| message (volatile status started)."""
        if len(msg.args) < 2:
            return
        from src.environment.protocol import parse_pokemon_ident

        ident = parse_pokemon_ident(msg.args[0])
        effect = msg.args[1].strip()
        if not ident:
            return

        if self._is_our_pokemon(ident) and self.state.own_active:
            self.state.own_active.volatiles.add(effect)
        elif self._is_opponent_pokemon(ident) and self.state.opponent_active:
            self.state.opponent_active.volatiles.add(effect)

    def _handle_end_effect(self, msg: BattleMessage) -> None:
        """Handle |-end| message (volatile status ended)."""
        if len(msg.args) < 2:
            return
        from src.environment.protocol import parse_pokemon_ident

        ident = parse_pokemon_ident(msg.args[0])
        effect = msg.args[1].strip()
        if not ident:
            return

        if self._is_our_pokemon(ident) and self.state.own_active:
            self.state.own_active.volatiles.discard(effect)
        elif self._is_opponent_pokemon(ident) and self.state.opponent_active:
            self.state.opponent_active.volatiles.discard(effect)

    def _handle_win(self, msg: BattleMessage) -> None:
        """Handle |win| message."""
        self.state.phase = GamePhase.FINISHED
        if msg.args:
            winner_name = msg.args[0].strip()
            # Resolve the username to a player ID (p1/p2) using the mapping
            # built from |player| messages.
            resolved = self.state._player_names.get(winner_name, winner_name)
            self.state.winner = resolved

    def _handle_tie(self, msg: BattleMessage) -> None:
        """Handle |tie| message."""
        self.state.phase = GamePhase.FINISHED
        self.state.winner = ""

    def _handle_cant(self, msg: BattleMessage) -> None:
        """Handle |cant| message (Pokemon can't act)."""
        pass  # Informational; state already managed by other messages

    def _handle_request(self, msg: BattleMessage) -> None:
        """Handle |request| message."""
        if msg.args:
            self.update_from_request(msg.args[0])

    def _handle_upkeep(self, msg: BattleMessage) -> None:
        """Handle |upkeep| message — decrement timed conditions."""
        # Decrement screen/tailwind/trick room counters
        for side in (self.state.own_side, self.state.opponent_side):
            for attr in _TIMED_CONDITIONS:
                val = getattr(side, attr, 0)
                if val > 0:
                    setattr(side, attr, val - 1)
        if self.state.field.trick_room > 0:
            self.state.field.trick_room -= 1
        if self.state.field.gravity > 0:
            self.state.field.gravity -= 1


# Map message types to handler methods
_MESSAGE_HANDLERS: dict[MessageType, Any] = {
    MessageType.PLAYER: BattleStateTracker._handle_player,
    MessageType.POKE: BattleStateTracker._handle_poke,
    MessageType.TEAMPREVIEW: BattleStateTracker._handle_teampreview,
    MessageType.START: BattleStateTracker._handle_start,
    MessageType.TURN: BattleStateTracker._handle_turn,
    MessageType.SWITCH: BattleStateTracker._handle_switch,
    MessageType.DRAG: BattleStateTracker._handle_switch,
    MessageType.MOVE: BattleStateTracker._handle_move,
    MessageType.DAMAGE: BattleStateTracker._handle_damage,
    MessageType.HEAL: BattleStateTracker._handle_heal,
    MessageType.FAINT: BattleStateTracker._handle_faint,
    MessageType.STATUS: BattleStateTracker._handle_status,
    MessageType.CURESTATUS: BattleStateTracker._handle_curestatus,
    MessageType.BOOST: BattleStateTracker._handle_boost,
    MessageType.UNBOOST: BattleStateTracker._handle_unboost,
    MessageType.CLEARBOOST: BattleStateTracker._handle_clearboost,
    MessageType.CLEARALLBOOST: BattleStateTracker._handle_clearboost,
    MessageType.WEATHER: BattleStateTracker._handle_weather,
    MessageType.FIELDSTART: BattleStateTracker._handle_fieldstart,
    MessageType.FIELDEND: BattleStateTracker._handle_fieldend,
    MessageType.SIDESTART: BattleStateTracker._handle_sidestart,
    MessageType.SIDEEND: BattleStateTracker._handle_sideend,
    MessageType.ITEM: BattleStateTracker._handle_item,
    MessageType.ENDITEM: BattleStateTracker._handle_enditem,
    MessageType.ABILITY: BattleStateTracker._handle_ability,
    MessageType.START_EFFECT: BattleStateTracker._handle_start_effect,
    MessageType.END_EFFECT: BattleStateTracker._handle_end_effect,
    MessageType.WIN: BattleStateTracker._handle_win,
    MessageType.TIE: BattleStateTracker._handle_tie,
    MessageType.CANT: BattleStateTracker._handle_cant,
    MessageType.REQUEST: BattleStateTracker._handle_request,
    MessageType.UPKEEP: BattleStateTracker._handle_upkeep,
}
