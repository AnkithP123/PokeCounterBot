import unittest
from unittest.mock import AsyncMock, MagicMock
import discord

from src.bot import (
    bot,
    find_target_channel,
    cmd_status,
    cmd_rules,
    cmd_check,
    cmd_set_count,
    cmd_set_number,
    cmd_classify,
    cmd_test,
    cmd_run_tests,
    validate_test_target,
    test_lock,
    get_game_for_channel,
    TARGET_CHANNEL_NAME
)


class TestSlashCommands(unittest.TestCase):
    def test_tree_commands_registered(self):
        """Verify that slash commands status, count_status, rules, check, set_count, set_number, classify, test, and run_tests are registered."""
        command_names = [cmd.name for cmd in bot.tree.get_commands()]
        self.assertIn("status", command_names)
        self.assertIn("count_status", command_names)
        self.assertIn("rules", command_names)
        self.assertIn("check", command_names)
        self.assertIn("set_count", command_names)
        self.assertIn("set_number", command_names)
        self.assertIn("classify", command_names)
        self.assertIn("test", command_names)
        self.assertIn("run_tests", command_names)

    def test_legacy_text_command_removed(self):
        """Verify that !count_status is no longer registered as a prefix command."""
        command_names = [cmd.name for cmd in bot.commands]
        self.assertNotIn("count_status", command_names)
        self.assertNotIn("cp_status", command_names)

    def test_find_target_channel(self):
        """Verify find_target_channel identifies the counter channel in a guild."""
        guild = MagicMock(spec=discord.Guild)
        ch1 = MagicMock(spec=discord.TextChannel)
        ch1.name = "general"
        ch2 = MagicMock(spec=discord.TextChannel)
        ch2.name = TARGET_CHANNEL_NAME
        guild.text_channels = [ch1, ch2]

        found = find_target_channel(guild)
        self.assertEqual(found, ch2)

        # Non-matching guild
        guild_empty = MagicMock(spec=discord.Guild)
        guild_empty.text_channels = [ch1]
        self.assertIsNone(find_target_channel(guild_empty))
        self.assertIsNone(find_target_channel(None))

    def test_cmd_status_success(self):
        """Verify /status replies with an embed containing current and next CP."""
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        guild = MagicMock(spec=discord.Guild)
        guild.name = "PokeGuild"

        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 99999
        channel.name = TARGET_CHANNEL_NAME
        channel.mention = f"<#{channel.id}>"
        channel.guild = guild

        interaction.guild = guild
        interaction.channel = channel

        # Set game state
        game = get_game_for_channel(channel.id)
        game.current_cp = 25
        game.starting_cp = 10

        import asyncio
        asyncio.run(cmd_status.callback(interaction))

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args[1]
        self.assertIn("embed", call_kwargs)
        embed = call_kwargs["embed"]
        self.assertEqual(embed.title, "📊 PokéCounter Status")
        field_dict = {f.name: f.value for f in embed.fields}
        self.assertEqual(field_dict["Current Count"], "`25`")
        self.assertEqual(field_dict["Next Expected CP"], "`26`")
        self.assertEqual(field_dict["Reset Base CP"], "`10`")

    def test_cmd_rules(self):
        """Verify /rules sends the rules embed."""
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        import asyncio
        asyncio.run(cmd_rules.callback(interaction))

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args[1]
        self.assertIn("embed", call_kwargs)
        embed = call_kwargs["embed"]
        self.assertEqual(embed.title, "📖 PokéCounter Game Rules")

    def test_cmd_check_non_image(self):
        """Verify /check rejects non-image attachments."""
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        attachment = MagicMock(spec=discord.Attachment)
        attachment.content_type = "text/plain"
        attachment.filename = "notes.txt"

        import asyncio
        asyncio.run(cmd_check.callback(interaction, attachment, visible=False))

        interaction.response.send_message.assert_called_once()
        self.assertIn("valid image", interaction.response.send_message.call_args[0][0])

    def test_cmd_set_count_valid(self):
        """Verify /set_count updates the current count and broadcasts {number} ✅ when user has manage_messages."""
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        user = MagicMock(spec=discord.Member)
        user.mention = "<@123>"
        interaction.user = user

        guild = MagicMock(spec=discord.Guild)
        guild.name = "PokeGuild"

        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 11111
        channel.name = TARGET_CHANNEL_NAME
        channel.guild = guild
        channel.send = AsyncMock()
        channel.permissions_for.return_value.manage_messages = True

        interaction.guild = guild
        interaction.channel = channel

        game = get_game_for_channel(channel.id)

        import asyncio
        asyncio.run(cmd_set_count.callback(interaction, number=50))

        self.assertEqual(game.current_cp, 50)
        self.assertEqual(game.next_expected_cp, 51)
        interaction.response.send_message.assert_called_once()
        msg = interaction.response.send_message.call_args[0][0]
        self.assertTrue(msg.startswith("50 ✅"))

    def test_cmd_set_count_permission_denied(self):
        """Verify /set_count rejects users who do not have permission to manage messages in the channel."""
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        user = MagicMock(spec=discord.Member)
        user.mention = "<@123>"
        interaction.user = user

        guild = MagicMock(spec=discord.Guild)
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 11112
        channel.name = TARGET_CHANNEL_NAME
        channel.mention = "<#11112>"
        channel.guild = guild
        channel.permissions_for.return_value.manage_messages = False

        interaction.guild = guild
        interaction.channel = channel

        import asyncio
        asyncio.run(cmd_set_count.callback(interaction, number=50))

        interaction.response.send_message.assert_called_once()
        self.assertIn("You don't have permission to do this.", interaction.response.send_message.call_args[0][0])

    def test_cmd_set_count_reset_zero(self):
        """Verify /set_count 0 resets the counter back to starting CP."""
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        user = MagicMock(spec=discord.Member)
        user.mention = "<@123>"
        interaction.user = user

        guild = MagicMock(spec=discord.Guild)
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 22222
        channel.name = TARGET_CHANNEL_NAME
        channel.guild = guild
        channel.permissions_for.return_value.manage_messages = True

        interaction.guild = guild
        interaction.channel = channel

        game = get_game_for_channel(channel.id)
        game.current_cp = 99

        import asyncio
        asyncio.run(cmd_set_count.callback(interaction, number=0))

        self.assertIsNone(game.current_cp)
        self.assertEqual(game.next_expected_cp, game.starting_cp)
        interaction.response.send_message.assert_called_once()
        msg = interaction.response.send_message.call_args[0][0]
        self.assertIn("0 ❌", msg)

    def test_cmd_set_count_invalid(self):
        """Verify /set_count rejects numbers outside 10-6000 (except 0)."""
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        guild = MagicMock(spec=discord.Guild)
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 33333
        channel.name = TARGET_CHANNEL_NAME
        channel.guild = guild
        channel.permissions_for.return_value.manage_messages = True

        interaction.guild = guild
        interaction.channel = channel

        import asyncio
        asyncio.run(cmd_set_count.callback(interaction, number=5))

        interaction.response.send_message.assert_called_once()
        self.assertIn("Invalid CP", interaction.response.send_message.call_args[0][0])

    def test_cmd_classify_non_image(self):
        """Verify /classify rejects non-image attachments."""
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        attachment = MagicMock(spec=discord.Attachment)
        attachment.content_type = "text/plain"
        attachment.filename = "report.txt"

        import asyncio
        asyncio.run(cmd_classify.callback(interaction, attachment, visible=False))

        interaction.response.send_message.assert_called_once()
        self.assertIn("valid image", interaction.response.send_message.call_args[0][0])

    def test_format_cp_display_references(self):
        """Verify format_cp_display formats references in parentheses correctly."""
        from src.bot import format_cp_display
        self.assertEqual(format_cp_display(69), "69 (nice)")
        self.assertEqual(format_cp_display(67), "67 (SIX SEVEN)")
        self.assertEqual(format_cp_display(42), "42 (the answer)")
        self.assertEqual(format_cp_display(404), "404 (not found)")
        self.assertEqual(format_cp_display(666), "666 (😈)")
        self.assertEqual(format_cp_display(777), "777 (jackpot!)")
        # Verify removed references
        self.assertEqual(format_cp_display(420), "420")
        self.assertEqual(format_cp_display(999), "999")
        self.assertEqual(format_cp_display(151), "151")
        self.assertEqual(format_cp_display(251), "251")
        self.assertEqual(format_cp_display(1025), "1025")
        self.assertEqual(format_cp_display(1337), "1337")
        self.assertEqual(format_cp_display(50), "50")

    def test_species_emoji_info(self):
        """Verify get_species_emoji_info maps requested Pokémon to their custom emojis."""
        from src.bot import get_species_emoji_info
        expected_mappings = {
            "Eevee": ":EeveeHeart:",
            "Machamp": ":FlexChamp:",
            "Mr. Mime": ":GASP:",
            "Chikorita": ":HUHH:",
            "Charmander": ":INFERNO:",
            "Rowlet": ":NotLikeThis:",
            "Grookey": ":ONLIFESUPPORT:",
            "Grimer": ":Pog:",
            "Munchlax": ":Popcorn:",
            "Sylveon": ":STARSTRUCK:",
            "Pikachu": ":Shocked:",
            "Wigglytuff": ":TARGETSIGHTED:",
            "Jigglypuff": ":TRIGGERED:",
            "Psyduck": ":WOW:",
            "Bidoof": ":bidoofCryLaugh:",
            "Buizel": ":buizelPing:",
            "Fuecoco": ":hehe:",
            "Pachirisu": ":pachiThinking:",
            "Politoed": ":politoadHands:",
            "Scorbunny": ":scoreThumbsUp:",
            "Seel": ":sips:",
            "Slowpoke": ":squint:",
            "Diglett": ":sus:",
            "Togekiss": ":togeSmile:",
            "Umbreon": ":umbreally:",
            "Wailmer": ":wailmDep:",
            "Wobbuffet": ":wob7:",
            "Wooper": ":woo:",
            "Ditto": ":dittoDance:",
            "Bellsprout": ":bellDance:",
            "Hitmontop": ":breakitdown:",
            "Spinda": ":spinda:",
            "Popplio": ":PopplioDance:",
            "Spheal": ":ROLLIN:",
            "Gimmighoul": ":GREED:",
        }
        for poke, expected_emoji in expected_mappings.items():
            _, emoji_str = get_species_emoji_info(poke)
            self.assertEqual(emoji_str, expected_emoji, f"Mismatch for {poke}")

    def test_milestone_suffix(self):
        """Verify get_milestone_suffix adds tada for 100, 200, 250-500 (by 50), and 1000, 1500, 2000, 2500, 3000, 4000, 5000."""
        from src.bot import get_milestone_suffix, is_milestone_cp
        self.assertTrue(is_milestone_cp(100))
        self.assertTrue(is_milestone_cp(200))
        self.assertTrue(is_milestone_cp(250))
        self.assertTrue(is_milestone_cp(300))
        self.assertTrue(is_milestone_cp(350))
        self.assertTrue(is_milestone_cp(400))
        self.assertTrue(is_milestone_cp(450))
        self.assertTrue(is_milestone_cp(500))

        # Exclude 600-900
        self.assertFalse(is_milestone_cp(550))
        self.assertFalse(is_milestone_cp(600))
        self.assertFalse(is_milestone_cp(700))
        self.assertFalse(is_milestone_cp(800))
        self.assertFalse(is_milestone_cp(900))
        self.assertFalse(is_milestone_cp(950))

        # Include major milestones
        self.assertTrue(is_milestone_cp(1000))
        self.assertTrue(is_milestone_cp(1500))
        self.assertTrue(is_milestone_cp(2000))
        self.assertTrue(is_milestone_cp(2500))
        self.assertTrue(is_milestone_cp(3000))
        self.assertTrue(is_milestone_cp(4000))
        self.assertTrue(is_milestone_cp(5000))

        self.assertFalse(is_milestone_cp(10))
        self.assertFalse(is_milestone_cp(50))
        self.assertFalse(is_milestone_cp(99))
        self.assertFalse(is_milestone_cp(101))

        self.assertEqual(get_milestone_suffix(100), " 🎉")
        self.assertEqual(get_milestone_suffix(250), " 🎉")
        self.assertEqual(get_milestone_suffix(1000), " 🎉")
        self.assertEqual(get_milestone_suffix(600), "")
        self.assertEqual(get_milestone_suffix(69), "")

    def test_validate_test_target(self):
        """Verify validate_test_target allows safe paths and blocks flags/traversal."""
        self.assertEqual(validate_test_target(None), "tests")
        self.assertEqual(validate_test_target(""), "tests")
        self.assertEqual(validate_test_target("tests"), "tests")
        self.assertEqual(validate_test_target("tests/test_slash_commands.py"), "tests/test_slash_commands.py")
        self.assertEqual(
            validate_test_target("tests/test_slash_commands.py::TestSlashCommands"),
            "tests/test_slash_commands.py::TestSlashCommands"
        )
        # Rejections
        self.assertIsNone(validate_test_target("-v"))
        self.assertIsNone(validate_test_target("../src/bot.py"))
        self.assertIsNone(validate_test_target("tests/../src/bot.py"))
        self.assertIsNone(validate_test_target("src/bot.py"))
        self.assertIsNone(validate_test_target("tests/test_slash_commands.py::bad;rm -rf /"))
        self.assertIsNone(validate_test_target("tests/nonexistent_test_file.py"))

    def test_cmd_test_invalid_target(self):
        """Verify /test rejects invalid target arguments."""
        import asyncio
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        asyncio.run(cmd_test.callback(interaction, target="-v"))
        interaction.response.send_message.assert_called_once()
        self.assertIn("Invalid test target", interaction.response.send_message.call_args[0][0])

    def test_cmd_test_already_locked(self):
        """Verify /test warns when a test suite is already executing."""
        import asyncio
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        async def run_when_locked():
            await test_lock.acquire()
            try:
                await cmd_test.callback(interaction, target=None)
            finally:
                test_lock.release()

        asyncio.run(run_when_locked())
        interaction.response.send_message.assert_called_once()
        self.assertIn("already in progress", interaction.response.send_message.call_args[0][0])

    def test_cmd_test_success_mock(self):
        """Verify /test runs subprocess, creates embed, and sends followup."""
        import asyncio
        from unittest.mock import patch

        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.defer = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"62 passed in 8.5s\n", b""))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            asyncio.run(cmd_test.callback(interaction, target="tests/test_slash_commands.py", visible=False))

        interaction.response.defer.assert_called_once_with(ephemeral=True)
        interaction.followup.send.assert_called_once()
        call_kwargs = interaction.followup.send.call_args[1]
        self.assertIn("embed", call_kwargs)
        embed = call_kwargs["embed"]
        self.assertEqual(embed.title, "🧪 Test Suite Results")
        self.assertEqual(embed.color, discord.Color.green())
        self.assertEqual(call_kwargs.get("ephemeral"), True)

    def test_cmd_test_failure_mock(self):
        """Verify /test displays red embed when tests fail."""
        import asyncio
        from unittest.mock import patch

        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.defer = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"1 failed, 61 passed\n", b""))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            asyncio.run(cmd_test.callback(interaction, target=None, visible=True))

        interaction.response.defer.assert_called_once_with(ephemeral=False)
        interaction.followup.send.assert_called_once()
        call_kwargs = interaction.followup.send.call_args[1]
        embed = call_kwargs["embed"]
        self.assertEqual(embed.color, discord.Color.red())
        self.assertEqual(call_kwargs.get("ephemeral"), False)

    def test_cmd_test_long_output_attachment(self):
        """Verify /test attaches a file when test output exceeds 1800 characters."""
        import asyncio
        from unittest.mock import patch

        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.defer = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        long_output = b"test line\n" * 300  # ~3000 bytes
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(long_output, b""))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            asyncio.run(cmd_test.callback(interaction, target=None, visible=False))

        interaction.followup.send.assert_called_once()
        call_kwargs = interaction.followup.send.call_args[1]
        self.assertIn("file", call_kwargs)
        self.assertIsInstance(call_kwargs["file"], discord.File)
        self.assertEqual(call_kwargs["file"].filename, "pytest_results.txt")

    def test_cmd_run_tests_alias(self):
        """Verify /run_tests aliases /test."""
        import asyncio
        from unittest.mock import patch

        interaction = MagicMock(spec=discord.Interaction)
        with patch.object(cmd_test, "_callback", AsyncMock()) as mock_cmd_test:
            asyncio.run(cmd_run_tests.callback(interaction, target="tests", visible=True))
            mock_cmd_test.assert_called_once_with(interaction, target="tests", visible=True)


if __name__ == "__main__":
    unittest.main()
