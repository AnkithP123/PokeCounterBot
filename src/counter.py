import re
from typing import Optional, Tuple


class PokeCounterGame:
    """
    Manages the Pokemon GO CP counting game state and rules.
    Can recover its state directly from the bot's own last message in the channel.
    """

    def __init__(self, starting_cp: int = 10, allow_consecutive_counts: bool = True):
        self.starting_cp = starting_cp
        self.allow_consecutive_counts = allow_consecutive_counts
        self.current_cp: Optional[int] = None
        self.last_user_id: Optional[int] = None

    @property
    def next_expected_cp(self) -> int:
        """Returns the next required CP number."""
        if self.current_cp is None:
            return self.starting_cp
        return self.current_cp + 1

    def parse_last_bot_message(self, content: str) -> Optional[int]:
        """
        Parses a past bot message to determine what count it recorded.
        Returns:
            - int N if the message was '{N} ✅'
            - None if the message was a reset (contained ❌) or unrecognized
        """
        if not content:
            return None

        # Check for reset / error message first
        if "❌" in content:
            return None

        # Check for CP format: e.g. "Pikachu CP 11", "Unknown Species CP 11", "✨ CP 11"
        match_cp = re.search(r'\bCP\s*(\d+)\b', content.strip(), re.IGNORECASE)
        if match_cp:
            return int(match_cp.group(1))

        # Check for legacy success pattern: e.g. "10 ✅", "10 ✅ (Pikachu)", or "⏳ 10 ✅"
        match_legacy = re.search(r'(?:^|[^\d])(\d+)\s*✅', content.strip())
        if match_legacy:
            return int(match_legacy.group(1))

        return None

    def recover_from_history(self, bot_messages: list) -> None:
        """
        Recovers the current game state from a list of past messages sent by the bot
        (ordered latest first).
        """
        self.current_cp = None
        self.last_user_id = None

        for msg in bot_messages:
            # We check the content of the bot's messages
            content = getattr(msg, "content", str(msg))
            parsed = self.parse_last_bot_message(content)
            if parsed is not None:
                self.current_cp = parsed
                return
            elif "❌" in content:
                # The latest bot message was a reset, so next is starting_cp
                self.current_cp = None
                return

        # If no relevant bot messages found, count starts at starting_cp
        self.current_cp = None

    def process_count(self, user_id: int, extracted_cp: int) -> Tuple[bool, str]:
        """
        Evaluates an extracted CP against the game rules.

        Returns:
            (is_correct, response_message)
        """
        is_consecutive = (self.last_user_id is not None and user_id == self.last_user_id)

        # Check consecutive count rule if disallowed
        if not self.allow_consecutive_counts and is_consecutive:
            self.current_cp = None
            self.last_user_id = None
            msg = f"{extracted_cp} ❌ You cannot count twice in a row! Restart at {self.starting_cp}."
            return False, msg

        expected = self.next_expected_cp

        if extracted_cp == expected:
            # Correct count!
            self.current_cp = extracted_cp
            self.last_user_id = user_id

            if is_consecutive and self.allow_consecutive_counts:
                return True, f"{extracted_cp} ✅\n⚠️ *Notice: Counting twice in a row will be disabled in the future.*"
            return True, f"{extracted_cp} ✅"
        else:
            # Wrong count! Reset state
            prev_cp = self.current_cp
            self.current_cp = None
            self.last_user_id = None
            if prev_cp is not None and extracted_cp == prev_cp:
                msg = f"{extracted_cp} ❌ CP {extracted_cp} was already counted! Expected next was {expected}. Resetting to {self.starting_cp}."
            else:
                msg = f"{extracted_cp} ❌ Wrong CP (expected {expected})! Begin at {self.starting_cp}."
            return False, msg
