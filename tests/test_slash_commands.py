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
    get_game_for_channel,
    TARGET_CHANNEL_NAME
)


class TestSlashCommands(unittest.TestCase):
    def test_tree_commands_registered(self):
        """Verify that slash commands status, count_status, rules, check, set_count, and set_number are registered."""
        command_names = [cmd.name for cmd in bot.tree.get_commands()]
        self.assertIn("status", command_names)
        self.assertIn("count_status", command_names)
        self.assertIn("rules", command_names)
        self.assertIn("check", command_names)
        self.assertIn("set_count", command_names)
        self.assertIn("set_number", command_names)

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
        """Verify /set_count updates the current count and broadcasts {number} ✅."""
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.user = MagicMock()
        interaction.user.mention = "<@123>"

        guild = MagicMock(spec=discord.Guild)
        guild.name = "PokeGuild"

        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 11111
        channel.name = TARGET_CHANNEL_NAME
        channel.guild = guild
        channel.send = AsyncMock()

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

    def test_cmd_set_count_reset_zero(self):
        """Verify /set_count 0 resets the counter back to starting CP."""
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.user = MagicMock()
        interaction.user.mention = "<@123>"

        guild = MagicMock(spec=discord.Guild)
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 22222
        channel.name = TARGET_CHANNEL_NAME
        channel.guild = guild

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

        interaction.guild = guild
        interaction.channel = channel

        import asyncio
        asyncio.run(cmd_set_count.callback(interaction, number=5))

        interaction.response.send_message.assert_called_once()
        self.assertIn("Invalid CP", interaction.response.send_message.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
