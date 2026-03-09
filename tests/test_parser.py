"""Tests for the Showdown replay parser.

Covers:
- Parsing of all event types (|switch|, |move|, |-damage|, |-status|, |faint|, etc.)
- Per-turn state extraction
- First-person reconstruction correctness
- Malformed/incomplete log handling

These tests will be implemented in Phase 2 (Step 2.2).
"""


class TestReplayParser:
    """Test Showdown log parsing."""

    def test_placeholder(self) -> None:
        """Placeholder — real tests added in Phase 2."""
        pass
