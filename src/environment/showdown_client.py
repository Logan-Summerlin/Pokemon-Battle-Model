"""WebSocket client for communicating with a local Pokemon Showdown server.

Handles:
- Connection management (connect, disconnect, reconnect)
- Authentication (guest login for local server)
- Battle room management (create, join, leave)
- Message sending and receiving
- Protocol message routing to the battle state tracker
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import websockets
import websockets.client

from src.environment.protocol import BattleMessage, MessageType, parse_battle_chunk

logger = logging.getLogger(__name__)


@dataclass
class ShowdownConfig:
    """Configuration for connecting to a Showdown server."""

    host: str = "localhost"
    port: int = 8000
    protocol: str = "ws"

    # Authentication
    username: str = ""
    password: str = ""  # Empty for guest login

    # Timeouts
    connect_timeout: float = 10.0
    message_timeout: float = 30.0
    battle_timeout: float = 300.0  # 5 minutes max per battle

    # Battle settings
    format: str = "gen3ou"
    team: str = ""  # Team in packed format

    @property
    def url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}/showdown/websocket"


@dataclass
class BattleRoom:
    """Represents a battle room on the server."""

    room_id: str = ""
    player_id: str = ""  # "p1" or "p2"
    opponent_name: str = ""
    format: str = ""
    is_active: bool = True
    messages: list[BattleMessage] = field(default_factory=list)


class ShowdownClient:
    """Async WebSocket client for Pokemon Showdown.

    Usage:
        client = ShowdownClient(config)
        await client.connect()
        await client.login("BotName")
        room = await client.challenge("opponent", format="gen3ou", team=packed_team)
        # ... receive messages and send actions ...
        await client.disconnect()
    """

    def __init__(self, config: ShowdownConfig | None = None) -> None:
        self.config = config or ShowdownConfig()
        self._ws: websockets.client.WebSocketClientProtocol | None = None
        self._rooms: dict[str, BattleRoom] = {}
        self._message_handlers: list[Callable[[str, list[BattleMessage]], Any]] = []
        self._connected = False
        self._challstr = ""
        self._username = ""
        self._raw_buffer: list[str] = []  # Buffer for unconsumed raw messages

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    def add_message_handler(
        self, handler: Callable[[str, list[BattleMessage]], Any]
    ) -> None:
        """Register a handler called for each batch of messages.

        Handler signature: handler(room_id: str, messages: list[BattleMessage])
        """
        self._message_handlers.append(handler)

    async def connect(self) -> None:
        """Establish WebSocket connection to the Showdown server."""
        try:
            self._ws = await asyncio.wait_for(
                websockets.client.connect(self.config.url),
                timeout=self.config.connect_timeout,
            )
            self._connected = True
            logger.info("Connected to Showdown server at %s", self.config.url)
        except (OSError, asyncio.TimeoutError) as e:
            logger.error("Failed to connect to Showdown server: %s", e)
            raise ConnectionError(
                f"Cannot connect to Showdown server at {self.config.url}: {e}"
            ) from e

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False
        self._rooms.clear()
        logger.info("Disconnected from Showdown server")

    async def login(self, username: str) -> None:
        """Log in as a guest user (for local server).

        On a local server with --no-security, guest login requires sending
        the /trn command with an empty assertion (no challstr needed).
        """
        self._username = username

        # Wait for the server to send initial data (including challstr)
        # even though we won't use the challstr for local no-security login
        if not self._challstr:
            await self._wait_for_challstr()

        # For local server with --no-security: |/trn username,0,
        # The empty assertion after the comma signals guest login
        # _send_global prepends "|", so we send just "/trn ..."
        await self._send_global(f"/trn {username},0,")

        # Wait for the server to confirm the name change
        deadline = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < deadline:
            raw = await self._receive_raw(timeout=2.0)
            if raw and "|updateuser|" in raw:
                logger.info("Logged in as %s", username)
                return
        logger.warning("Login response not confirmed for %s, proceeding anyway", username)

    async def _wait_for_challstr(self, timeout: float = 10.0) -> None:
        """Wait for the server to send the challstr."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            raw = await self._receive_raw(timeout=1.0)
            if raw is None:
                continue
            for line in raw.split("\n"):
                if "|challstr|" in line:
                    parts = line.split("|challstr|")
                    if len(parts) >= 2:
                        self._challstr = parts[1].strip()
                        return
        raise TimeoutError("Did not receive challstr from server")

    async def search_battle(self, format: str, team: str = "") -> str:
        """Start searching for a battle.

        Args:
            format: Battle format (e.g., "gen3ou")
            team: Team in packed/export format

        Returns:
            The room ID once a battle is found.
        """
        if team:
            await self._send_global(f"/utm {team}")
        await self._send_global(f"/search {format}")
        logger.info("Searching for %s battle...", format)

        # Wait for battle to be found
        room_id = await self._wait_for_battle_room()
        return room_id

    async def challenge(
        self, opponent: str, format: str = "gen3ou", team: str = ""
    ) -> str:
        """Challenge a specific user to a battle.

        Returns the room ID once the challenge is accepted.
        """
        if team:
            await self._send_global(f"/utm {team}")
        await self._send_global(f"/challenge {opponent}, {format}")
        logger.info("Challenging %s to %s...", opponent, format)
        room_id = await self._wait_for_battle_room()
        return room_id

    async def accept_challenge(self, challenger: str, team: str = "") -> str:
        """Accept a pending challenge.

        Returns the room ID.
        """
        if team:
            await self._send_global(f"/utm {team}")
        await self._send_global(f"/accept {challenger}")
        room_id = await self._wait_for_battle_room()
        return room_id

    async def send_choice(self, room_id: str, choice: str) -> None:
        """Send a battle choice (move/switch) to a room.

        Args:
            room_id: The battle room ID
            choice: Choice string (e.g., "/choose move 1", "/choose switch 3")
        """
        if not choice.startswith("/choose"):
            choice = f"/choose {choice}"
        await self._send_room(room_id, choice)

    async def send_team_order(self, room_id: str, order: str = "default") -> None:
        """Send team order during team preview.

        Args:
            room_id: The battle room ID
            order: Team order string (e.g., "123456" for default order)
        """
        await self._send_room(room_id, f"/choose team {order}")

    async def forfeit(self, room_id: str) -> None:
        """Forfeit the current battle."""
        await self._send_room(room_id, "/forfeit")

    async def leave_room(self, room_id: str) -> None:
        """Leave a battle room."""
        await self._send_room(room_id, "/leave")
        self._rooms.pop(room_id, None)

    async def receive_messages(
        self, timeout: float | None = None
    ) -> tuple[str, list[BattleMessage]]:
        """Receive and parse the next batch of messages.

        Returns:
            Tuple of (room_id, parsed_messages). Room ID is empty for global messages.
        """
        timeout = timeout or self.config.message_timeout

        # Drain the buffer first (messages stored by _wait_for_battle_room)
        if self._raw_buffer:
            raw = self._raw_buffer.pop(0)
        else:
            raw = await self._receive_raw(timeout=timeout)
        if raw is None:
            return "", []

        # Parse room ID from the first line
        room_id = ""
        lines = raw.strip().split("\n")
        if lines and lines[0].startswith(">"):
            room_id = lines[0][1:].strip()
            raw = "\n".join(lines[1:])

        messages = parse_battle_chunk(raw)

        # Track in room
        if room_id and room_id in self._rooms:
            self._rooms[room_id].messages.extend(messages)

        # Notify handlers
        for handler in self._message_handlers:
            handler(room_id, messages)

        return room_id, messages

    async def receive_until(
        self,
        predicate: Callable[[str, list[BattleMessage]], bool],
        timeout: float | None = None,
    ) -> list[tuple[str, list[BattleMessage]]]:
        """Receive messages until a predicate is satisfied.

        Args:
            predicate: Function(room_id, messages) -> bool. Stop when True.
            timeout: Maximum time to wait.

        Returns:
            All message batches received.
        """
        timeout = timeout or self.config.battle_timeout
        all_batches: list[tuple[str, list[BattleMessage]]] = []
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break

            try:
                room_id, messages = await self.receive_messages(timeout=remaining)
            except asyncio.TimeoutError:
                break

            if messages:
                all_batches.append((room_id, messages))
                if predicate(room_id, messages):
                    break

        return all_batches

    # ── Private methods ─────────────────────────────────────────────────

    async def _send_global(self, message: str) -> None:
        """Send a message to the global (lobby) context."""
        if not self._ws:
            raise ConnectionError("Not connected to server")
        await self._ws.send(f"|{message}")

    async def _send_room(self, room_id: str, message: str) -> None:
        """Send a message to a specific room."""
        if not self._ws:
            raise ConnectionError("Not connected to server")
        await self._ws.send(f"{room_id}|{message}")

    async def _receive_raw(self, timeout: float = 10.0) -> str | None:
        """Receive a raw message from the server."""
        if not self._ws:
            return None
        try:
            return await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except websockets.exceptions.ConnectionClosed:
            self._connected = False
            return None

    async def _wait_for_battle_room(self, timeout: float = 30.0) -> str:
        """Wait for a battle room to be created.

        Any received messages that contain battle data are stored in
        ``_raw_buffer`` so that they are not lost.  The next call to
        ``receive_messages`` will drain the buffer first.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            raw = await self._receive_raw(timeout=1.0)
            if raw is None:
                continue
            lines = raw.strip().split("\n")
            room_id = ""
            if lines and lines[0].startswith(">battle-"):
                room_id = lines[0][1:].strip()
            else:
                for line in lines:
                    if "|init|battle" in line or "|title|" in line:
                        if lines[0].startswith(">"):
                            room_id = lines[0][1:].strip()
                        break

            if room_id:
                self._rooms[room_id] = BattleRoom(room_id=room_id)
                # Store the raw message so the battle stream can parse it
                self._raw_buffer.append(raw)
                return room_id
            # Non-battle messages (PMs, updatesearch) can be discarded
        raise TimeoutError("Battle room not created within timeout")


class ShowdownBattleStream:
    """High-level interface for playing a single battle.

    Wraps ShowdownClient to provide a simpler, turn-by-turn interface
    for bot integration.
    """

    def __init__(self, client: ShowdownClient, room_id: str) -> None:
        self.client = client
        self.room_id = room_id
        self._pending_messages: list[BattleMessage] = []
        self._battle_ended = False

    @property
    def is_active(self) -> bool:
        return not self._battle_ended

    async def receive_until_request(self) -> list[BattleMessage]:
        """Receive messages until we get a |request| or the battle ends.

        Returns all messages received (including the request).
        """
        all_messages: list[BattleMessage] = list(self._pending_messages)
        self._pending_messages.clear()

        def has_request_or_end(
            room_id: str, messages: list[BattleMessage]
        ) -> bool:
            if room_id != self.room_id:
                return False
            for msg in messages:
                if msg.msg_type in (
                    MessageType.REQUEST,
                    MessageType.WIN,
                    MessageType.TIE,
                ):
                    return True
            return False

        batches = await self.client.receive_until(has_request_or_end)
        for room_id, messages in batches:
            if room_id == self.room_id:
                all_messages.extend(messages)
            else:
                # Store messages for other rooms
                self._pending_messages.extend(messages)

        # Check if battle ended
        for msg in all_messages:
            if msg.msg_type in (MessageType.WIN, MessageType.TIE):
                self._battle_ended = True

        return all_messages

    async def send_action(self, choice: str) -> None:
        """Send a battle action."""
        await self.client.send_choice(self.room_id, choice)

    async def send_team_order(self, order: str = "default") -> None:
        """Send team order for team preview."""
        await self.client.send_team_order(self.room_id, order)
