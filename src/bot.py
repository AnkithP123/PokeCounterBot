import os
import sys
import asyncio
import logging
from typing import Optional, Dict, Set, Any

import discord
from discord.ext import commands
from dotenv import load_dotenv

from src.ocr import extract_cp_from_image
from src.counter import PokeCounterGame
from src.classifier import classify_pokemon_from_image, check_hp_validity_for_candidate

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
synced_channels: Set[int] = set()
active_stat_corrections: Dict[int, Any] = {}


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


async def sync_game_state_from_channel(
    channel: discord.TextChannel,
    before_message: Optional[discord.Message] = None
) -> PokeCounterGame:
    """Recovers the current game state by inspecting the bot's past messages in the channel."""
    game = get_game_for_channel(channel.id)
    guild_name = channel.guild.name if channel.guild else "Unknown"
    bot_messages = []
    has_history = False
    try:
        if hasattr(channel, "history"):
            history_kwargs: Dict[str, Any] = {"limit": 100}
            if before_message is not None:
                history_kwargs["before"] = before_message

            history_iter = channel.history(**history_kwargs)
            if hasattr(history_iter, "__aiter__"):
                has_history = True
                async for msg in history_iter:
                    if getattr(msg, "author", None) and msg.author.id == bot.user.id:
                        bot_messages.append(msg)

        if bot_messages:
            game.recover_from_history(bot_messages)
            synced_channels.add(channel.id)
            logger.info(
                "Synced [%s] #%s from chat: Current CP: %s, Next expected CP: %s",
                guild_name,
                channel.name,
                game.current_cp,
                game.next_expected_cp
            )
    except Exception as e:
        logger.error("Failed to read channel history for [%s] #%s: %s", guild_name, channel.name, e)

    return game


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


def get_particle_placeholder() -> str:
    """
    Returns the placeholder emoji:
    Prioritizes :slow: as requested, or PARTICLES_EMOJI / LOADING_EMOJI env var,
    or emojis matching 'slow', 'particle', 'sparkle', 'star', 'loading', or 'spinner'.
    Falls back to ✨.
    """
    env_emoji = os.getenv("PARTICLES_EMOJI") or os.getenv("LOADING_EMOJI")
    if env_emoji:
        return env_emoji

    # 1. Prioritize :slow: emoji (case-insensitive exact or partial match)
    for emoji in bot.emojis:
        if emoji.name.lower() == "slow":
            return str(emoji)
    for emoji in bot.emojis:
        if "slow" in emoji.name.lower():
            return str(emoji)

    # 2. Search for animated emojis matching keywords (excluding dance/dittodance)
    keywords = ["particle", "sparkle", "star", "loading", "spinner"]
    for kw in keywords:
        for emoji in bot.emojis:
            if getattr(emoji, "animated", False) and kw in emoji.name.lower():
                return str(emoji)

    # Any animated emoji if none matched the keywords (excluding dance)
    for emoji in bot.emojis:
        if getattr(emoji, "animated", False) and "dance" not in emoji.name.lower():
            return str(emoji)

    return "✨"


class StatCorrectionView(discord.ui.View):
    def __init__(self, bot, target_message: discord.Message, reply_msg: discord.Message,
                 channel_id: int, user_id: int, extracted_cp: int, detected_hp: Optional[int],
                 family_display: str, consecutive_warning: str, previous_cp: Optional[int],
                 previous_last_user_id: Optional[int]):
        super().__init__(timeout=300)
        self.bot = bot
        self.target_message = target_message
        self.reply_msg = reply_msg
        self.correction_msg: Optional[discord.Message] = None
        self.channel_id = channel_id
        self.user_id = user_id
        self.extracted_cp = extracted_cp
        self.detected_hp = detected_hp
        self.family_display = family_display
        self.consecutive_warning = consecutive_warning
        self.previous_cp = previous_cp
        self.previous_last_user_id = previous_last_user_id
        self.resolved = False

    async def cleanup(self):
        if not self.resolved:
            self.resolved = True
            self.stop()
            if self.correction_msg:
                try:
                    await self.correction_msg.delete()
                except Exception:
                    pass

    @discord.ui.button(label="Enter Real HP", style=discord.ButtonStyle.primary, emoji="✏️")
    async def btn_enter_hp(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            is_mod = False
            if isinstance(interaction.user, discord.Member):
                is_mod = interaction.user.guild_permissions.manage_messages
            if not is_mod:
                await interaction.response.send_message(
                    "❌ Only the person who uploaded this screenshot can correct the HP.",
                    ephemeral=True
                )
                return

        modal = EnterHpModal(self)
        await interaction.response.send_modal(modal)

    async def on_timeout(self):
        if not self.resolved:
            self.resolved = True
            for item in self.children:
                item.disabled = True
            if self.correction_msg:
                try:
                    await self.correction_msg.edit(view=self)
                except Exception:
                    pass


class EnterHpModal(discord.ui.Modal, title="Verify Pokémon HP"):
    def __init__(self, view: StatCorrectionView):
        super().__init__()
        self.parent_view = view
        default_val = str(view.detected_hp) if view.detected_hp is not None else ""
        self.hp_input = discord.ui.TextInput(
            label="Actual HP (from screenshot)",
            placeholder="e.g. 35",
            default=default_val,
            required=True,
            min_length=1,
            max_length=4
        )
        self.add_item(self.hp_input)

    async def on_submit(self, interaction: discord.Interaction):
        val = self.hp_input.value.strip()
        try:
            entered_hp = int(val)
        except ValueError:
            await interaction.response.send_message(
                f"❌ '{val}' is not a valid number. Please enter a valid HP integer.",
                ephemeral=True
            )
            return

        is_valid, valid_species, family_disp = check_hp_validity_for_candidate(
            self.parent_view.family_display,
            self.parent_view.extracted_cp,
            entered_hp
        )

        display_name = family_disp or self.parent_view.family_display
        game = get_game_for_channel(self.parent_view.channel_id)

        if is_valid:
            self.parent_view.resolved = True
            self.parent_view.stop()
            active_stat_corrections.pop(self.parent_view.channel_id, None)

            species_text = valid_species or display_name
            final_text = f"{species_text} CP {self.parent_view.extracted_cp} ✅{self.parent_view.consecutive_warning}"
            game.current_cp = self.parent_view.extracted_cp
            game.last_user_id = self.parent_view.user_id

            # 1. Change back to a check
            try:
                await self.parent_view.target_message.remove_reaction("🤨", self.parent_view.bot.user)
            except Exception:
                pass
            try:
                await self.parent_view.target_message.add_reaction("✅")
            except Exception:
                pass

            # 2. Edit the original message
            try:
                await self.parent_view.reply_msg.edit(content=final_text, view=None)
            except Exception as e:
                logger.warning("Could not edit message after HP fix: %s", e)

            # 3. Make the other message disappear
            if self.parent_view.correction_msg:
                try:
                    await self.parent_view.correction_msg.delete()
                except Exception as e:
                    logger.warning("Could not delete correction message: %s", e)

            await interaction.response.send_message(
                f"✅ HP verified as **{entered_hp}**! Updated count to **{species_text} CP {self.parent_view.extracted_cp}**.",
                ephemeral=True
            )
        else:
            # If they enter a still impossible one:
            # Don't change the reaction or original message.
            # Delete the message with the interaction button, keeping only the ephemeral message.
            self.parent_view.resolved = True
            self.parent_view.stop()
            active_stat_corrections.pop(self.parent_view.channel_id, None)

            if self.parent_view.correction_msg:
                try:
                    await self.parent_view.correction_msg.delete()
                except Exception as e:
                    logger.warning("Could not delete correction message on invalid HP: %s", e)

            await interaction.response.send_message(
                f"❌ CP {self.parent_view.extracted_cp} with HP {entered_hp} is still mathematically impossible for {display_name}. Please upload an unedited screenshot.",
                ephemeral=True
            )


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

    # Always determine the current count and next expected CP directly from the chat
    game = await sync_game_state_from_channel(message.channel, before_message=message)

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
        logger.error("Error reading image attachment: %s", e)
        return

    if extracted_cp is None:
        logger.info(
            "No CP detected in image from %s in [%s] #%s",
            message.author.name,
            message.guild.name if message.guild else "",
            message.channel.name
        )
        await message.add_reaction("❓")
        await message.reply(
            f"❓ Could not detect CP. Next expected CP is still {game.next_expected_cp}.",
            mention_author=False
        )
        return

    logger.info(
        "Detected CP: %d from %s in [%s] #%s (Expected: %d)",
        extracted_cp,
        message.author.name,
        message.guild.name if message.guild else "",
        message.channel.name,
        game.next_expected_cp
    )

    previous_cp = game.current_cp
    previous_last_user_id = game.last_user_id
    is_correct, response_text = game.process_count(message.author.id, extracted_cp)

    if is_correct:
        # Clean up any lingering active stat correction in this channel
        if message.channel.id in active_stat_corrections:
            prior_corr = active_stat_corrections.pop(message.channel.id)
            asyncio.create_task(prior_corr.cleanup())

        await message.add_reaction("✅")
        # Build initial response with placeholder and checkmark at the end: e.g. ":slow: CP 11 ✅"
        consecutive_warning = "\n⚠️ *Notice: Counting twice in a row will be disabled in the future.*" if "⚠️" in response_text else ""
        particle = get_particle_placeholder()
        initial_text = f"{particle} CP {extracted_cp} ✅{consecutive_warning}"
        reply_msg = await message.reply(initial_text, mention_author=False)

        # Classify species asynchronously in background and update the message
        async def update_with_species():
            try:
                async with message.channel.typing():
                    res = await asyncio.to_thread(classify_pokemon_from_image, image_bytes, known_cp=extracted_cp)
                    species = res.get("species")
                    final_cp = extracted_cp
                    stat_status = res.get("stat_status", "VALID")
                    detected_hp = res.get("hp")
                    family_display = res.get("family_display") or species or "This Pokémon"
            except Exception as ex:
                logger.error("Error classifying species on count: %s", ex)
                species = None
                final_cp = extracted_cp
                stat_status = "VALID"
                detected_hp = None
                family_display = "Unknown Species"

            if stat_status == "IMPOSSIBLE":
                # Impossible Pokémon NEVER resets the count and does not advance it
                cur_game = get_game_for_channel(message.channel.id)
                if cur_game.current_cp == extracted_cp:
                    cur_game.current_cp = previous_cp
                    cur_game.last_user_id = previous_last_user_id

                # Change checkmark reaction to raised eyebrow emoji
                try:
                    await message.remove_reaction("✅", bot.user)
                except Exception as ex:
                    logger.warning("Could not remove checkmark reaction: %s", ex)
                try:
                    await message.add_reaction("🤨")
                except Exception as ex:
                    logger.warning("Could not add raised eyebrow reaction: %s", ex)

                # Edit initial message:
                # Warning sign at beginning, says it's impossible saying to upload unedited,
                # at end lets them know the number is still that number
                warning_text = (
                    f"⚠️ Impossible Pokémon: CP {extracted_cp} with HP {detected_hp} is mathematically impossible for {family_display}. "
                    f"Please upload an unedited screenshot. "
                    f"Next expected CP is still {extracted_cp}."
                )
                try:
                    await reply_msg.edit(content=warning_text, view=None)
                except Exception as edit_err:
                    logger.warning("Could not edit initial count message with warning: %s", edit_err)

                # Reply to that message with button to enter correct HP
                view = StatCorrectionView(
                    bot=bot,
                    target_message=message,
                    reply_msg=reply_msg,
                    channel_id=message.channel.id,
                    user_id=message.author.id,
                    extracted_cp=extracted_cp,
                    detected_hp=detected_hp,
                    family_display=family_display,
                    consecutive_warning=consecutive_warning,
                    previous_cp=previous_cp,
                    previous_last_user_id=previous_last_user_id
                )
                active_stat_corrections[message.channel.id] = view

                correction_text = (
                    f"{message.author.mention} If the HP in your screenshot was misread, "
                    f"click below to enter the correct HP from your screenshot:"
                )
                try:
                    correction_msg = await reply_msg.reply(
                        correction_text,
                        view=view,
                        mention_author=True
                    )
                    view.correction_msg = correction_msg
                except Exception as reply_err:
                    logger.warning("Could not send correction reply: %s", reply_err)
                return

            if species:
                species_name = species
            else:
                species_name = "Unknown Species"

            final_text = f"{species_name} CP {final_cp} ✅{consecutive_warning}"
            try:
                await reply_msg.edit(content=final_text)
            except Exception as edit_err:
                logger.warning("Could not edit count message with species: %s", edit_err)

        asyncio.create_task(update_with_species())
    else:
        await message.add_reaction("❌")
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

    game = await sync_game_state_from_channel(target_channel)
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
        game = await sync_game_state_from_channel(target_channel)
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


@bot.tree.command(name="classify", description="Identify the Pokémon species from a Pokémon GO screenshot.")
@discord.app_commands.describe(
    image="The Pokémon GO screenshot to analyze",
    visible="Whether to make the result visible to everyone in the channel (default: False)"
)
async def cmd_classify(interaction: discord.Interaction, image: discord.Attachment, visible: bool = False):
    is_img = (image.content_type and image.content_type.startswith("image/")) or image.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    if not is_img:
        await interaction.response.send_message("❌ Please upload a valid image file (.png, .jpg, .webp).", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=not visible)

    try:
        image_bytes = await image.read()
        res = await asyncio.to_thread(classify_pokemon_from_image, image_bytes)
    except Exception as e:
        logger.error("Error classifying image in /classify: %s", e)
        await interaction.followup.send("⚠️ Failed to process image file.", ephemeral=not visible)
        return

    species = res.get("species")
    cp = res.get("cp")
    hp = res.get("hp")
    candy = res.get("candy_family")
    explanation = res.get("explanation", "")
    stardust = res.get("stardust")
    candy_count = res.get("candy_count")
    powerup_dust = res.get("powerup_stardust")
    powerup_candy = res.get("powerup_candy")
    level = res.get("estimated_level")
    ivs = res.get("appraisal_ivs")

    if not species:
        desc = "⚠️ **Could not identify the Pokémon species.**\n"
        if not candy:
            desc += "Make sure the screenshot shows the stats card with the **Candy** section clearly visible (not an appraisal screen or cropped view)."
        else:
            desc += f"Found `{candy}` Candy, but could not determine the exact species.\n*{explanation}*"
        embed = discord.Embed(title="🔎 Pokémon Classifier", description=desc, color=discord.Color.red())
        embed.set_thumbnail(url=image.url)
        await interaction.followup.send(embed=embed, ephemeral=not visible)
        return

    embed = discord.Embed(
        title="🔎 Pokémon Species Identified",
        description=f"Species: **{species}**",
        color=discord.Color.green()
    )
    if cp is not None:
        embed.add_field(name="CP", value=f"`{cp}`", inline=True)
    if hp is not None:
        embed.add_field(name="HP", value=f"`{hp}`", inline=True)
    if candy:
        embed.add_field(name="Candy Family", value=f"`{candy}`", inline=True)
    if stardust is not None:
        embed.add_field(name="Stardust", value=f"⭐ `{stardust:,}`", inline=True)
    if candy_count is not None:
        embed.add_field(name="Candy Stock", value=f"🍬 `{candy_count}`", inline=True)
    if powerup_dust is not None:
        pu_str = f"`{powerup_dust:,} Dust`"
        if powerup_candy:
            pu_str += f" + `{powerup_candy} Candy`"
        if level:
            pu_str += f" *(Lvl {level})*"
        embed.add_field(name="Power Up", value=pu_str, inline=True)
    if ivs:
        embed.add_field(
            name="Appraisal IVs",
            value=f"**{ivs['percent']}%** (Atk: `{ivs['atk']}`, Def: `{ivs['def']}`, HP: `{ivs['sta']}`)",
            inline=False
        )
    if explanation:
        embed.set_footer(text=explanation)

    embed.set_thumbnail(url=image.url)
    await interaction.followup.send(embed=embed, ephemeral=not visible)



def main():
    if not TOKEN:
        logger.critical("No DISCORD_TOKEN found in environment or .env file!")
        sys.exit(1)

    bot.run(TOKEN)


if __name__ == "__main__":
    main()
