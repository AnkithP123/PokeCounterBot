import unittest
from src.counter import PokeCounterGame


class TestPokeCounterGame(unittest.TestCase):
    def test_initial_state(self):
        game = PokeCounterGame(starting_cp=10)
        self.assertIsNone(game.current_cp)
        self.assertEqual(game.next_expected_cp, 10)

    def test_successful_progression(self):
        game = PokeCounterGame(starting_cp=10, allow_consecutive_counts=True)

        # First count: 10
        ok, msg = game.process_count(user_id=1, extracted_cp=10)
        self.assertTrue(ok)
        self.assertEqual(msg, "10 ✅")
        self.assertEqual(game.current_cp, 10)
        self.assertEqual(game.next_expected_cp, 11)

        # Second count: 11 by user 2
        ok, msg = game.process_count(user_id=2, extracted_cp=11)
        self.assertTrue(ok)
        self.assertEqual(msg, "11 ✅")
        self.assertEqual(game.current_cp, 11)
        self.assertEqual(game.next_expected_cp, 12)

        # Third count: 12 by user 2 again (consecutive) -> succeeds with warning notice
        ok, msg = game.process_count(user_id=2, extracted_cp=12)
        self.assertTrue(ok)
        self.assertIn("12 ✅", msg)
        self.assertIn("⚠️ *Notice: Counting twice in a row will be disabled in the future.*", msg)
        self.assertEqual(game.current_cp, 12)
        self.assertEqual(game.next_expected_cp, 13)

    def test_wrong_number_resets_game(self):
        game = PokeCounterGame(starting_cp=10)

        # Initial 10
        game.process_count(user_id=1, extracted_cp=10)
        self.assertEqual(game.next_expected_cp, 11)

        # Wrong count: 13 instead of 11
        ok, msg = game.process_count(user_id=2, extracted_cp=13)
        self.assertFalse(ok)
        self.assertIn("13 ❌ Wrong CP, should have been 11. Begin at 10.", msg)
        self.assertIsNone(game.current_cp)
        self.assertEqual(game.next_expected_cp, 10)

    def test_consecutive_count_restriction(self):
        game = PokeCounterGame(starting_cp=10, allow_consecutive_counts=False)

        # User 1 sends 10
        ok, msg = game.process_count(user_id=1, extracted_cp=10)
        self.assertTrue(ok)

        # User 1 tries to send 11 consecutively
        ok, msg = game.process_count(user_id=1, extracted_cp=11)
        self.assertFalse(ok)
        self.assertIn("cannot count twice in a row", msg)
        self.assertEqual(game.next_expected_cp, 10)

    def test_history_recovery_from_success(self):
        game = PokeCounterGame(starting_cp=10)

        class FakeMessage:
            def __init__(self, content):
                self.content = content

        history = [
            FakeMessage("15 ✅"),
            FakeMessage("14 ✅"),
        ]

        game.recover_from_history(history)
        self.assertEqual(game.current_cp, 15)
        self.assertEqual(game.next_expected_cp, 16)

    def test_history_recovery_with_species_and_placeholders(self):
        class FakeMessage:
            def __init__(self, content):
                self.content = content

        test_cases = [
            ("10 ✅", 10),
            ("15 ✅", 15),
            ("15 ✅ *(Count manually set by <@12345> — next expected CP is 16)*", 15),
            ("Pikachu CP 11", 11),
            ("Unknown Species CP 11", 11),
            ("<a:particles:123456789> CP 11", 11),
            ("✨ CP 11", 11),
            ("Pikachu CP 11\n⚠️ *Notice: Counting twice in a row will be disabled in the future.*", 11),
            ("✨ CP 42\n⚠️ *Notice: Counting twice in a row will be disabled in the future.*", 42),
            ("15 ✅ (Pikachu)", 15),
            ("15 ✅ (Unknown species)", 15),
            ("⏳ 15 ✅ *(Identifying species...)*", 15),
            ("15 ✅ (Charizard)\n⚠️ *Notice: Counting twice in a row will be disabled in the future.*", 15),
            ("⏳ 42 ✅ *(Identifying species...)*\n⚠️ *Notice: Counting twice in a row will be disabled in the future.*", 42),
            ("<a:slow:999999> CP 11 ✅", 11),
            ("Pikachu CP 11 ✅", 11),
            ("Unknown Species CP 11 ✅", 11),
            ("Ditto CP 10 ✅", 10),
            ("Ditto CP 10 (edited)", 10),
            ("<a:DittoDance:88888> CP 10", 10),
            ("<a:slow:999999> CP 11 ✅\n⚠️ *Notice: Counting twice in a row will be disabled in the future.*", 11),
            ("CP: 25", 25),
            ("30 CP", 30),
        ]

        for text, expected_cp in test_cases:
            game = PokeCounterGame(starting_cp=10)
            game.recover_from_history([FakeMessage(text)])
            self.assertEqual(game.current_cp, expected_cp, f"Failed on content: {text}")
            self.assertEqual(game.next_expected_cp, expected_cp + 1)

    def test_history_recovery_from_reset(self):
        game = PokeCounterGame(starting_cp=10)

        class FakeMessage:
            def __init__(self, content):
                self.content = content

        history = [
            FakeMessage("18 ❌ Wrong CP, begin at 10."),
            FakeMessage("17 ✅"),
        ]

        game.recover_from_history(history)
        self.assertIsNone(game.current_cp)
        self.assertEqual(game.next_expected_cp, 10)

    def test_history_recovery_empty(self):
        game = PokeCounterGame(starting_cp=10)
        game.recover_from_history([])
        self.assertIsNone(game.current_cp)
        self.assertEqual(game.next_expected_cp, 10)

    def test_history_recovery_skips_unknown_cp_warnings(self):
        """Verify that unknown CP warning messages with ❓ are skipped and previous count is preserved."""
        game = PokeCounterGame(starting_cp=10)

        class FakeMessage:
            def __init__(self, content):
                self.content = content

        history = [
            FakeMessage("❓ Could not detect CP. Next expected CP is still 16."),
            FakeMessage("15 ✅"),
        ]

        game.recover_from_history(history)
        self.assertEqual(game.current_cp, 15)
        self.assertEqual(game.next_expected_cp, 16)

    def test_history_recovery_skips_impossible_pokemon_warnings(self):
        """Verify that impossible Pokemon warning and correction messages are skipped and do not reset count."""
        game = PokeCounterGame(starting_cp=10)

        class FakeMessage:
            def __init__(self, content):
                self.content = content

        history = [
            FakeMessage("❌ <@123> CP 68 with HP 109 is still mathematically impossible for Bunnelby / Diggersby. Please upload an unedited screenshot."),
            FakeMessage("⚠️ Impossible Pokémon: CP 68 with HP 109 is mathematically impossible for Bunnelby / Diggersby. Please upload an unedited screenshot. Next expected CP is still 68."),
            FakeMessage("67 ✅"),
        ]

        game.recover_from_history(history)
        self.assertEqual(game.current_cp, 67)
        self.assertEqual(game.next_expected_cp, 68)


    def test_history_recovery_from_chat_overrides_stale_memory(self):
        """Verify that recovering from chat history sets the exact CP from the chat."""
        game = PokeCounterGame(starting_cp=10)
        # Pretend memory had a stale or incorrect value
        game.current_cp = 999

        class FakeMessage:
            def __init__(self, content):
                self.content = content

        history = [
            FakeMessage("Pikachu CP 42 ✅"),
            FakeMessage("Bulbasaur CP 41 ✅"),
        ]

        game.recover_from_history(history)
        self.assertEqual(game.current_cp, 42)
        self.assertEqual(game.next_expected_cp, 43)

    def test_parse_reference_and_milestone_messages(self):
        """Verify that parse_last_bot_message correctly extracts CP from messages with references and milestone tada."""
        game = PokeCounterGame(starting_cp=10)
        self.assertEqual(game.parse_last_bot_message("Pikachu CP 67 (SIX SEVEN) ✅"), 67)
        self.assertEqual(game.parse_last_bot_message("Pikachu CP 69 (nice) ✅"), 69)
        self.assertEqual(game.parse_last_bot_message("Pikachu CP 100 ✅ 🎉"), 100)
        self.assertEqual(game.parse_last_bot_message("Pikachu CP 250 ✅ 🎉"), 250)
        self.assertEqual(game.parse_last_bot_message("Pikachu CP 300 ✅ 🎉"), 300)
        self.assertEqual(game.parse_last_bot_message(":slow: CP 69 (nice) ✅"), 69)
        self.assertEqual(game.parse_last_bot_message(":slow: CP 100 ✅ 🎉"), 100)
        self.assertEqual(game.parse_last_bot_message("Mew CP 151 (Kanto complete!) ✅"), 151)
        self.assertEqual(game.parse_last_bot_message("Porygon CP 404 (not found) ✅"), 404)
        self.assertEqual(game.parse_last_bot_message("Oddish CP 420 (blaze it) ✅"), 420)


if __name__ == "__main__":
    unittest.main()


