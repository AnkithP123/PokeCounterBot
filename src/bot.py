import os
import sys
import asyncio
import logging
from typing import Optional

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
game = PokeCounterGame(starting_cp=STARTING_CP, allow_consecutive_counts=ALLOW_CONSECUTIVE)


def is_target_channel(channel: discord.abc.GuildChannel) -> bool:
    """Checks whether the channel is the configured #poke-counter channel."""
    if TARGET_CHANNEL_ID is not None:
        return channel.id == TARGET_CHANNEL_ID
    return getattr(channel, "name", "").lower() == TARGET_CHANNEL_NAME


async def sync_game_state_from_channel(channel: discord.TextChannel) -> None:
    """Recovers the current game state by inspecting the bot's past messages in the channel."""
    logger.info("Syncing game state from channel #%s (ID: %s)...", channel.name, channel.id)
    bot_messages = []
    try:
        async for msg in channel.history(limit=60):
            if msg.author.id == bot.user.id:
                bot_messages.append(msg)
        
        game.recover_from_history(bot_messages)
        logger.info(
            "Game state synced successfully! Current CP: %s, Next expected CP: %s",
            game.current_cp,
            game.next_expected_cp
        )
    except Exception as e:
        logger.error("Failed to read channel history for state recovery: %s", e)


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

    # Sync state for all matching text channels
    found = False
    for guild in bot.guilds:
        for channel in guild.text_channels:
            if is_target_channel(channel):
                found = True
                await sync_game_state_from_channel(channel)
                break

    if not found:
        logger.warning(
            "Could not find target channel '%s' (or ID %s) in any connected servers!",
            TARGET_CHANNEL_NAME,
            TARGET_CHANNEL_ID
        )


@bot.event
async def on_message(message: discord.Message):
    # Ignore own messages or bot messages
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
        # User spoke in the channel without an image attachment
        await bot.process_commands(message)
        return

    # Process first image attachment
    attachment = image_attachments[0]
    logger.info(
        "Processing image from %s (file: %s, size: %d bytes)",
        message.author.name,
        attachment.filename,
        attachment.size
    )

    try:
        image_bytes = await attachment.read()
        # Run OCR in background thread so event loop is not blocked
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
        "Extracted CP: %d from user %s. Next expected: %d",
        extracted_cp,
        message.author.name,
        game.next_expected_cp
    )

    is_correct, response_text = game.process_count(message.author.id, extracted_cp)

    if is_correct:
        await message.add_reaction("✅")
    else:
        await message.add_reaction("❌")

    await message.channel.send(response_text)
    await bot.process_commands(message)


@bot.command(name="count_status", aliases=["cp_status", "count"])
async def cmd_count_status(ctx: commands.Context):
    """Replies with the current counting status."""
    if not is_target_channel(ctx.channel):
        return

    # Resync from history to guarantee freshness
    await sync_game_state_from_channel(ctx.channel)
    current = game.current_cp if game.current_cp is not None else "None (Game not started or reset)"
    expected = game.next_expected_cp
    await ctx.send(
        f"📊 **PokeCounter Status**\n"
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
