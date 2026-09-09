import os
import sys
import asyncio
import logging
from typing import Optional, Dict

import discord
from discord.ext import commands
from dotenv import load_dotenv

from src.ocr import extract_cp_from_image
from src.counter import PokeCounterGame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PokeCounterBot")

# Load environment variables
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
TARGET_CHANNEL_NAME = os.getenv("CHANNEL_NAME", "poke-counter").lower()
TARGET_CHANNEL_ID_STR = os.getenv("CHANNEL_ID")
TARGET_CHANNEL_ID: Optional[int] = int(TARGET_CHANNEL_ID_STR) if TARGET_CHANNEL_ID_STR else None
STARTING_CP = int(os.getenv("STARTING_CP", "10"))
ALLOW_CONSECUTIVE = os.getenv("ALLOW_CONSECUTIVE_COUNTS", "true").lower() in ("1", "true", "yes")

intents = discord.Intents.default()
intents.message_content = True
intents.guild_messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Multi-server support: Each channel maintains its own independent counting game
games: Dict[int, PokeCounterGame] = {}


def is_target_channel(channel: discord.abc.GuildChannel) -> bool:
    """Checks whether the channel is the configured #poke-counter channel."""
    if TARGET_CHANNEL_ID is not None:
        return channel.id == TARGET_CHANNEL_ID
    return getattr(channel, "name", "").lower() == TARGET_CHANNEL_NAME


def get_game_for_channel(channel_id: int) -> PokeCounterGame:
    """Gets or initializes the counting game state for a specific channel."""
    if channel_id not in games:
        games[channel_id] = PokeCounterGame(
            starting_cp=STARTING_CP,
            allow_consecutive_counts=ALLOW_CONSECUTIVE
        )
    return games[channel_id]


async def sync_game_state_from_channel(channel: discord.TextChannel) -> None:
    """Recovers the current game state by inspecting the bot's past messages in the channel."""
    game = get_game_for_channel(channel.id)
    guild_name = channel.guild.name if channel.guild else "Unknown"
    logger.info("Syncing game state for [%s] #%s (ID: %s)...", guild_name, channel.name, channel.id)
    bot_messages = []
    try:
        async for msg in channel.history(limit=60):
            if msg.author.id == bot.user.id:
                bot_messages.append(msg)

        game.recover_from_history(bot_messages)
        logger.info(
            "Synced [%s] #%s: Current CP: %s, Next expected CP: %s",
            guild_name,
            channel.name,
            game.current_cp,
            game.next_expected_cp
        )
    except Exception as e:
        logger.error("Failed to read channel history for [%s] #%s: %s", guild_name, channel.name, e)


@bot.event
async def on_ready():
    logger.info("Logged in as %s (ID: %s)", bot.user.name, bot.user.id)
    logger.info(
        "Configuration: Target channel='%s' (ID=%s), Starting CP=%d, Allow Consecutive=%s",
        TARGET_CHANNEL_NAME,
        TARGET_CHANNEL_ID,
        STARTING_CP,
        ALLOW_CONSECUTIVE
    )

    # Sync state for all matching text channels across all joined servers
    target_channels_found = 0
    for guild in bot.guilds:
        for channel in guild.text_channels:
            if is_target_channel(channel):
                target_channels_found += 1
                await sync_game_state_from_channel(channel)

    logger.info(
        "PokeCounterBot is ready across %d servers (monitoring %d #%s channels)",
        len(bot.guilds),
        target_channels_found,
        TARGET_CHANNEL_NAME
    )


@bot.event
async def on_guild_join(guild: discord.Guild):
    """When invited to a new server, automatically discover and sync any #poke-counter channel."""
    logger.info("Joined new guild: %s (ID: %s)", guild.name, guild.id)
    for channel in guild.text_channels:
        if is_target_channel(channel):
            await sync_game_state_from_channel(channel)


@bot.event
async def on_message(message: discord.Message):
    # Ignore own messages or other bot messages
    if message.author.bot or message.author == bot.user:
        return

    # Only process messages in the target channel
    if not is_target_channel(message.channel):
        await bot.process_commands(message)
        return

    # Check for image attachments
    image_attachments = [
        att for att in message.attachments
        if (att.content_type and att.content_type.startswith("image/"))
        or att.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    if not image_attachments:
        # User sent text without an image attachment
        await bot.process_commands(message)
        return

    attachment = image_attachments[0]
    game = get_game_for_channel(message.channel.id)

    logger.info(
        "Processing image from %s in [%s] #%s (file: %s, size: %d bytes)",
        message.author.name,
        message.guild.name if message.guild else "DM",
        message.channel.name,
        attachment.filename,
        attachment.size
    )

    try:
        async with message.channel.typing():
            image_bytes = await attachment.read()
            extracted_cp = await asyncio.to_thread(extract_cp_from_image, image_bytes)
    except Exception as e:
        logger.error("Error reading/processing image attachment: %s", e)
        await message.add_reaction("⚠️")
        return

    if extracted_cp is None:
        logger.warning("Could not extract CP from image sent by %s", message.author.name)
        await message.add_reaction("❓")
        await message.reply(
            "⚠️ Could not detect a Pokémon CP in that screenshot. Make sure the CP at the top is clearly visible!",
            mention_author=False
        )
        return

    logger.info(
        "[%s #%s] Extracted CP: %d from user %s. Next expected: %d",
        message.guild.name if message.guild else "",
        message.channel.name,
        extracted_cp,
        message.author.name,
        game.next_expected_cp
    )

    is_correct, response_text = game.process_count(message.author.id, extracted_cp)

    if is_correct:
        await message.add_reaction("✅")
    else:
        await message.add_reaction("❌")

    # Reply directly to the user's screenshot message without pinging them
    await message.reply(response_text, mention_author=False)
    await bot.process_commands(message)


@bot.command(name="count_status", aliases=["cp_status", "count"])
async def cmd_count_status(ctx: commands.Context):
    """Replies with the current counting status for this channel."""
    if not is_target_channel(ctx.channel):
        return

    await sync_game_state_from_channel(ctx.channel)
    game = get_game_for_channel(ctx.channel.id)
    current = game.current_cp if game.current_cp is not None else "None (Game not started or reset)"
    expected = game.next_expected_cp
    await ctx.send(
        f"📊 **PokeCounter Status**\n"
        f"• Server: **{ctx.guild.name}**\n"
        f"• Current Count: `{current}`\n"
        f"• Next Expected CP: `{expected}`\n"
        f"• Reset Base: `{game.starting_cp}`"
    )


def main():
    if not TOKEN:
        logger.critical("No DISCORD_TOKEN found in environment or .env file!")
        sys.exit(1)

    bot.run(TOKEN)


if __name__ == "__main__":
    main()
