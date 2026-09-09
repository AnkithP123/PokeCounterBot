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
        self.assertIn("13 ❌ Wrong CP, begin at 10.", msg)
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


if __name__ == "__main__":
    unittest.main()
