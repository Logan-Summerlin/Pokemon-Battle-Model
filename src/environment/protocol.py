"""Pokemon Showdown protocol message parser.

Parses the Showdown server's text protocol into structured Python objects.
Reference: https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md

Each message line starts with '|' followed by a message type and pipe-delimited arguments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(Enum):
    """Known Showdown protocol message types."""

    # Battle initialization
    PLAYER = "player"
    TEAMSIZE = "teamsize"
    GAMETYPE = "gametype"
    GEN = "gen"
    TIER = "tier"
    RULE = "rule"
    START = "start"
    CLEARPOKE = "clearpoke"
    POKE = "poke"
    TEAMPREVIEW = "teampreview"

    # Turn flow
    TURN = "turn"
    UPKEEP = "upkeep"
    REQUEST = "request"
    WIN = "win"
    TIE = "tie"

    # Battle actions
    MOVE = "move"
    SWITCH = "switch"
    DRAG = "drag"
    SWAP = "swap"
    CANT = "cant"
    FAINT = "faint"

    # Minor actions (prefixed with - in protocol)
    DAMAGE = "-damage"
    HEAL = "-heal"
    SETHP = "-sethp"
    STATUS = "-status"
    CURESTATUS = "-curestatus"
    CURETEAM = "-cureteam"
    BOOST = "-boost"
    UNBOOST = "-unboost"
    SETBOOST = "-setboost"
    CLEARBOOST = "-clearboost"
    CLEARALLBOOST = "-clearallboost"
    CLEARPOSITIVEBOOST = "-clearpositiveboost"
    CLEARNEGATIVEBOOST = "-clearnegativeboost"
    COPYBOOST = "-copyboost"
    WEATHER = "-weather"
    FIELDSTART = "-fieldstart"
    FIELDEND = "-fieldend"
    SIDESTART = "-sidestart"
    SIDEEND = "-sideend"
    ITEM = "-item"
    ENDITEM = "-enditem"
    ABILITY = "-ability"
    ENDABILITY = "-endability"
    TRANSFORM = "-transform"
    MEGA = "-mega"
    ACTIVATE = "-activate"
    PREPARE = "-prepare"
    MUSTRECHARGE = "-mustrecharge"
    SINGLETURN = "-singleturn"
    SINGLEMOVE = "-singlemove"
    START_EFFECT = "-start"
    END_EFFECT = "-end"
    CRIT = "-crit"
    SUPEREFFECTIVE = "-supereffective"
    RESISTED = "-resisted"
    IMMUNE = "-immune"
    MISS = "-miss"
    FAIL = "-fail"
    BLOCK = "-block"
    NOTARGET = "-notarget"
    OHKO = "-ohko"
    COMBINE = "-combine"
    HITCOUNT = "-hitcount"

    # Chat / misc
    CHAT = "c"
    CHATMSG = "c:"
    RAW = "raw"
    HTML = "html"
    INACTIVE = "inactive"
    INACTIVEOFF = "inactiveoff"
    TIMESTAMP = "t:"

    UNKNOWN = "unknown"


# ── Structured message dataclasses ──────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PokemonIdent:
    """Identifies a Pokemon in battle: player position + name."""

    player: str  # "p1" or "p2"
    position: str  # "a" for singles (could be "b" for doubles)
    name: str  # Display name (nickname or species)

    @property
    def player_num(self) -> int:
        return int(self.player[1])


@dataclass(frozen=True, slots=True)
class PokemonDetails:
    """Species details from a switch/drag/poke message."""

    species: str
    level: int = 100
    gender: str = ""  # "M", "F", or "" for genderless
    shiny: bool = False

    @property
    def base_species(self) -> str:
        """Get base species name (strips forme suffixes for display)."""
        return self.species.split("-")[0] if "-" in self.species else self.species


@dataclass(frozen=True, slots=True)
class HPStatus:
    """HP and status condition from protocol."""

    current_hp: int
    max_hp: int
    status: str = ""  # "brn", "par", "slp", "psn", "tox", "frz", or ""

    @property
    def fraction(self) -> float:
        if self.max_hp == 0:
            return 0.0
        return self.current_hp / self.max_hp

    @property
    def is_fainted(self) -> bool:
        return self.current_hp == 0


@dataclass(frozen=True, slots=True)
class BattleMessage:
    """A parsed protocol message."""

    msg_type: MessageType
    args: list[str] = field(default_factory=list)
    kwargs: dict[str, str] = field(default_factory=dict)
    raw: str = ""


# ── Parsing functions ───────────────────────────────────────────────────────

# Pattern for Pokemon ident: "p1a: Pikachu" or "p2a: Charizard"
_IDENT_RE = re.compile(r"^(p[12])([a-z]): (.+)$")

# Pattern for HP/status: "100/100", "50/100 brn", "0 fnt"
_HP_RE = re.compile(r"^(\d+)/(\d+)(?:\s+(\w+))?$")
_FAINT_RE = re.compile(r"^0\s+fnt$")

# Pattern for details: "Pikachu, L50, M" or "Charizard-Mega-X, L100, F, shiny"
_DETAILS_RE = re.compile(r"^([^,]+)(?:,\s*L(\d+))?(?:,\s*([MF]))?(?:,\s*(shiny))?$")


def parse_pokemon_ident(text: str) -> PokemonIdent | None:
    """Parse a Pokemon identifier like 'p1a: Pikachu'."""
    match = _IDENT_RE.match(text.strip())
    if not match:
        return None
    return PokemonIdent(
        player=match.group(1),
        position=match.group(2),
        name=match.group(3),
    )


def parse_pokemon_details(text: str) -> PokemonDetails:
    """Parse Pokemon details like 'Pikachu, L50, M'."""
    text = text.strip()
    match = _DETAILS_RE.match(text)
    if not match:
        return PokemonDetails(species=text)
    return PokemonDetails(
        species=match.group(1).strip(),
        level=int(match.group(2)) if match.group(2) else 100,
        gender=match.group(3) or "",
        shiny=match.group(4) is not None,
    )


def parse_hp_status(text: str) -> HPStatus:
    """Parse HP/status like '100/100', '50/100 brn', or '0 fnt'."""
    text = text.strip()
    if _FAINT_RE.match(text):
        return HPStatus(current_hp=0, max_hp=0, status="fnt")
    match = _HP_RE.match(text)
    if not match:
        return HPStatus(current_hp=0, max_hp=0)
    return HPStatus(
        current_hp=int(match.group(1)),
        max_hp=int(match.group(2)),
        status=match.group(3) or "",
    )


def _classify_message_type(type_str: str) -> MessageType:
    """Convert a raw message type string to MessageType enum."""
    try:
        return MessageType(type_str)
    except ValueError:
        return MessageType.UNKNOWN


def parse_message(line: str) -> BattleMessage:
    """Parse a single protocol message line.

    Lines have the format: |messagetype|arg1|arg2|...
    Some args contain [key] value pairs in kwargs format.
    """
    line = line.strip()
    if not line.startswith("|"):
        return BattleMessage(msg_type=MessageType.UNKNOWN, raw=line)

    parts = line[1:].split("|")
    if not parts:
        return BattleMessage(msg_type=MessageType.UNKNOWN, raw=line)

    type_str = parts[0]
    args = parts[1:]

    # Extract [key] value kwargs from args
    kwargs: dict[str, str] = {}
    clean_args: list[str] = []
    for arg in args:
        bracket_match = re.match(r"^\[(\w+)\]\s*(.*)$", arg.strip())
        if bracket_match:
            kwargs[bracket_match.group(1)] = bracket_match.group(2).strip()
        else:
            clean_args.append(arg)

    return BattleMessage(
        msg_type=_classify_message_type(type_str),
        args=clean_args,
        kwargs=kwargs,
        raw=line,
    )


def parse_battle_chunk(chunk: str) -> list[BattleMessage]:
    """Parse a multi-line chunk of battle messages.

    A chunk is typically what the server sends in one websocket frame,
    containing multiple newline-separated messages.
    """
    messages = []
    for line in chunk.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        messages.append(parse_message(line))
    return messages


# ── Specialized parsers for complex message types ───────────────────────────


def parse_switch_message(msg: BattleMessage) -> dict[str, Any]:
    """Extract structured data from a |switch| or |drag| message.

    Format: |switch|POKEMON|DETAILS|HP STATUS
    """
    if len(msg.args) < 3:
        return {}
    ident = parse_pokemon_ident(msg.args[0])
    details = parse_pokemon_details(msg.args[1])
    hp_status = parse_hp_status(msg.args[2])
    return {
        "ident": ident,
        "details": details,
        "hp_status": hp_status,
    }


def parse_move_message(msg: BattleMessage) -> dict[str, Any]:
    """Extract structured data from a |move| message.

    Format: |move|POKEMON|MOVE|TARGET
    Optional kwargs: [from], [of], [still], [miss]
    """
    result: dict[str, Any] = {"kwargs": msg.kwargs}
    if len(msg.args) >= 1:
        result["source"] = parse_pokemon_ident(msg.args[0])
    if len(msg.args) >= 2:
        result["move"] = msg.args[1].strip()
    if len(msg.args) >= 3:
        result["target"] = parse_pokemon_ident(msg.args[2])
    return result


def parse_damage_message(msg: BattleMessage) -> dict[str, Any]:
    """Extract structured data from a |-damage| or |-heal| message.

    Format: |-damage|POKEMON|HP STATUS
    """
    result: dict[str, Any] = {"kwargs": msg.kwargs}
    if len(msg.args) >= 1:
        result["target"] = parse_pokemon_ident(msg.args[0])
    if len(msg.args) >= 2:
        result["hp_status"] = parse_hp_status(msg.args[1])
    return result


def parse_boost_message(msg: BattleMessage) -> dict[str, Any]:
    """Extract structured data from a |-boost| or |-unboost| message.

    Format: |-boost|POKEMON|STAT|AMOUNT
    """
    result: dict[str, Any] = {"kwargs": msg.kwargs}
    if len(msg.args) >= 1:
        result["target"] = parse_pokemon_ident(msg.args[0])
    if len(msg.args) >= 2:
        result["stat"] = msg.args[1].strip()
    if len(msg.args) >= 3:
        try:
            result["amount"] = int(msg.args[2].strip())
        except ValueError:
            result["amount"] = 0
    return result


def parse_status_message(msg: BattleMessage) -> dict[str, Any]:
    """Extract structured data from a |-status| or |-curestatus| message.

    Format: |-status|POKEMON|STATUS
    """
    result: dict[str, Any] = {"kwargs": msg.kwargs}
    if len(msg.args) >= 1:
        result["target"] = parse_pokemon_ident(msg.args[0])
    if len(msg.args) >= 2:
        result["status"] = msg.args[1].strip()
    return result


def parse_weather_message(msg: BattleMessage) -> dict[str, Any]:
    """Extract structured data from a |-weather| message.

    Format: |-weather|WEATHER
    """
    result: dict[str, Any] = {"kwargs": msg.kwargs}
    if len(msg.args) >= 1:
        result["weather"] = msg.args[0].strip()
    return result


def parse_field_message(msg: BattleMessage) -> dict[str, Any]:
    """Extract structured data from |-fieldstart| or |-fieldend|.

    Format: |-fieldstart|CONDITION
    """
    result: dict[str, Any] = {"kwargs": msg.kwargs}
    if len(msg.args) >= 1:
        result["condition"] = msg.args[0].strip()
    return result


def parse_side_condition_message(msg: BattleMessage) -> dict[str, Any]:
    """Extract structured data from |-sidestart| or |-sideend|.

    Format: |-sidestart|SIDE|CONDITION
    """
    result: dict[str, Any] = {"kwargs": msg.kwargs}
    if len(msg.args) >= 1:
        # Side is like "p1: PlayerName"
        side_str = msg.args[0].strip()
        if ":" in side_str:
            result["player"] = side_str.split(":")[0].strip()
        else:
            result["player"] = side_str
    if len(msg.args) >= 2:
        result["condition"] = msg.args[1].strip()
    return result


def parse_item_message(msg: BattleMessage) -> dict[str, Any]:
    """Extract structured data from |-item| or |-enditem|.

    Format: |-item|POKEMON|ITEM
    """
    result: dict[str, Any] = {"kwargs": msg.kwargs}
    if len(msg.args) >= 1:
        result["target"] = parse_pokemon_ident(msg.args[0])
    if len(msg.args) >= 2:
        result["item"] = msg.args[1].strip()
    return result


def parse_ability_message(msg: BattleMessage) -> dict[str, Any]:
    """Extract structured data from |-ability| or |-endability|.

    Format: |-ability|POKEMON|ABILITY
    """
    result: dict[str, Any] = {"kwargs": msg.kwargs}
    if len(msg.args) >= 1:
        result["target"] = parse_pokemon_ident(msg.args[0])
    if len(msg.args) >= 2:
        result["ability"] = msg.args[1].strip()
    return result


def parse_poke_message(msg: BattleMessage) -> dict[str, Any]:
    """Extract structured data from |poke| (team preview).

    Format: |poke|PLAYER|DETAILS|ITEM
    """
    result: dict[str, Any] = {}
    if len(msg.args) >= 1:
        result["player"] = msg.args[0].strip()
    if len(msg.args) >= 2:
        result["details"] = parse_pokemon_details(msg.args[1])
    if len(msg.args) >= 3:
        result["has_item"] = msg.args[2].strip() == "item"
    else:
        result["has_item"] = False
    return result
