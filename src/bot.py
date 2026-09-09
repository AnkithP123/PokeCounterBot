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

class PokeCounterClient(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        try:
            synced = await self.tree.sync()
            logger.info("Synchronized %d application command(s) globally.", len(synced))
        except Exception as e:
            logger.error("Failed to sync application command tree: %s", e)


bot = PokeCounterClient()

# Multi-server support: Each channel maintains its own independent counting game
games: Dict[int, PokeCounterGame] = {}


def is_target_channel(channel: discord.abc.GuildChannel) -> bool:
    """Checks whether the channel is the configured #poke-counter channel."""
    if TARGET_CHANNEL_ID is not None:
        return channel.id == TARGET_CHANNEL_ID
    return getattr(channel, "name", "").lower() == TARGET_CHANNEL_NAME


def find_target_channel(guild: Optional[discord.Guild]) -> Optional[discord.TextChannel]:
    """Finds the configured #poke-counter channel within a guild."""
    if not guild:
        return None
    for channel in guild.text_channels:
        if is_target_channel(channel):
            return channel
    return None


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


@bot.tree.command(name="status", description="Check the current Pokémon counting game status and next required CP.")
async def cmd_status(interaction: discord.Interaction):
    guild = interaction.guild
    target_channel = None
    if is_target_channel(interaction.channel):
        target_channel = interaction.channel
    elif guild:
        target_channel = find_target_channel(guild)

    if not target_channel:
        await interaction.response.send_message(
            f"❌ Could not find a `#{TARGET_CHANNEL_NAME}` channel in this server.",
            ephemeral=True
        )
        return

    game = get_game_for_channel(target_channel.id)
    current_val = f"`{game.current_cp}`" if game.current_cp is not None else "*None (Game not started or reset)*"
    expected_val = f"`{game.next_expected_cp}`"

    embed = discord.Embed(
        title="📊 PokéCounter Status",
        color=discord.Color.blue()
    )
    embed.add_field(name="Channel", value=target_channel.mention, inline=True)
    embed.add_field(name="Current Count", value=current_val, inline=True)
    embed.add_field(name="Next Expected CP", value=expected_val, inline=True)
    embed.add_field(name="Reset Base CP", value=f"`{game.starting_cp}`", inline=True)
    if guild:
        embed.set_footer(text=f"Server: {guild.name}")

    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="count_status", description="Alias for /status. Check current count and next required CP.")
async def cmd_count_status(interaction: discord.Interaction):
    await cmd_status(interaction)


@bot.tree.command(name="set_count", description="Set the current Pokémon count number for this server.")
@discord.app_commands.default_permissions(manage_messages=True)
@discord.app_commands.describe(
    number="The new current CP count (10-6000, or 0 to reset to base starting CP)"
)
async def cmd_set_count(interaction: discord.Interaction, number: int):
    guild = interaction.guild
    target_channel = None
    if is_target_channel(interaction.channel):
        target_channel = interaction.channel
    elif guild:
        target_channel = find_target_channel(guild)

    if not target_channel:
        await interaction.response.send_message(
            f"❌ Could not find a `#{TARGET_CHANNEL_NAME}` channel in this server.",
            ephemeral=True
        )
        return

    # Restrict to members who have permission to delete/manage messages in the target channel
    user = interaction.user
    if isinstance(user, discord.Member):
        perms = target_channel.permissions_for(user)
        if not perms.manage_messages:
            await interaction.response.send_message(
                "❌ You don't have permission to do this.",
                ephemeral=True
            )
            return

    game = get_game_for_channel(target_channel.id)

    if number == 0:
        # Reset to base starting CP
        game.current_cp = None
        game.last_user_id = None
        reset_msg = f"0 ❌ *(Count reset by {interaction.user.mention} — next expected CP is {game.starting_cp})*"
        if interaction.channel.id == target_channel.id:
            await interaction.response.send_message(reset_msg)
        else:
            await target_channel.send(reset_msg)
            await interaction.response.send_message(
                f"🔄 Count in {target_channel.mention} has been reset. Next expected CP is `{game.starting_cp}`.",
                ephemeral=True
            )
        return

    if number < 10 or number > 6000:
        await interaction.response.send_message(
            f"❌ Invalid CP: `{number}`. Please enter a valid Pokémon GO CP between 10 and 6000 (or `0` to reset).",
            ephemeral=True
        )
        return

    game.current_cp = number
    game.last_user_id = None
    set_msg = f"{number} ✅ *(Count manually set by {interaction.user.mention} — next expected CP is {number + 1})*"

    if interaction.channel.id == target_channel.id:
        await interaction.response.send_message(set_msg)
    else:
        await target_channel.send(set_msg)
        await interaction.response.send_message(
            f"✅ Count in {target_channel.mention} set to `{number}`. Next expected CP is `{number + 1}`.",
            ephemeral=True
        )


@bot.tree.command(name="set_number", description="Alias for /set_count. Set the current count number.")
@discord.app_commands.default_permissions(manage_messages=True)
@discord.app_commands.describe(
    number="The new current CP count (10-6000, or 0 to reset to base starting CP)"
)
async def cmd_set_number(interaction: discord.Interaction, number: int):
    await cmd_set_count(interaction, number)



@bot.tree.command(name="rules", description="View the rules and how to play the Pokémon counting game.")
async def cmd_rules(interaction: discord.Interaction):
    embed = discord.Embed(
        title="📖 PokéCounter Game Rules",
        description="Work together with your server to count as high as possible using Pokémon GO screenshots!",
        color=discord.Color.green()
    )
    embed.add_field(
        name="1️⃣ How to Count",
        value=f"Find a Pokémon in Pokémon GO with the exact **Next Expected CP**, take a screenshot, and post it in the counting channel.",
        inline=False
    )
    embed.add_field(
        name="2️⃣ Automatic OCR Detection",
        value="The bot automatically scans your screenshot and reads the CP at the top. If correct, the count progresses (`✅`)!",
        inline=False
    )
    embed.add_field(
        name="3️⃣ Wrong Count / Reset",
        value=f"Posting an incorrect CP or broken count resets the game back to `{STARTING_CP}` (`❌`).",
        inline=False
    )
    embed.add_field(
        name="4️⃣ Take Turns",
        value="Please take turns! Counting twice in a row will be disabled in the future.",
        inline=False
    )
    embed.add_field(
        name="💡 Screenshot Tip",
        value="Ensure the `CP XXXX` banner near the top is clearly visible and not obscured by notifications or widgets.",
        inline=False
    )
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="check", description="Test CP detection on a Pokémon screenshot without affecting the count.")
@discord.app_commands.describe(
    image="The Pokémon GO screenshot to test",
    visible="Whether to make the result visible to everyone in the channel (default: False)"
)
async def cmd_check(interaction: discord.Interaction, image: discord.Attachment, visible: bool = False):
    is_img = (image.content_type and image.content_type.startswith("image/")) or image.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    if not is_img:
        await interaction.response.send_message("❌ Please upload a valid image file (.png, .jpg, .webp).", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=not visible)

    try:
        image_bytes = await image.read()
        extracted_cp = await asyncio.to_thread(extract_cp_from_image, image_bytes)
    except Exception as e:
        logger.error("Error reading image in /check: %s", e)
        await interaction.followup.send("⚠️ Failed to process image file.", ephemeral=not visible)
        return

    if extracted_cp is None:
        await interaction.followup.send(
            "⚠️ **Could not detect a Pokémon CP in that image.**\nMake sure the CP banner near the top is clearly visible and unobstructed.",
            ephemeral=not visible
        )
        return

    target_channel = None
    if is_target_channel(interaction.channel):
        target_channel = interaction.channel
    elif interaction.guild:
        target_channel = find_target_channel(interaction.guild)

    status_extra = ""
    if target_channel:
        game = get_game_for_channel(target_channel.id)
        if extracted_cp == game.next_expected_cp:
            status_extra = f"\n🎯 **Matches next expected CP ({game.next_expected_cp})!** Ready to count."
        else:
            status_extra = f"\nℹ️ *Current next expected CP in {target_channel.mention} is `{game.next_expected_cp}`.*"

    embed = discord.Embed(
        title="🔍 CP Scan Result",
        description=f"Detected CP: **{extracted_cp}**{status_extra}",
        color=discord.Color.gold()
    )
    embed.set_thumbnail(url=image.url)
    await interaction.followup.send(embed=embed, ephemeral=not visible)


def main():
    if not TOKEN:
        logger.critical("No DISCORD_TOKEN found in environment or .env file!")
        sys.exit(1)

    bot.run(TOKEN)


if __name__ == "__main__":
    main()
