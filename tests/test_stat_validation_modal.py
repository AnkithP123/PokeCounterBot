import cv2
import pytest
from src.classifier import (
    check_hp_validity_for_candidate,
    classify_pokemon_from_image
)
from src.bot import StatCorrectionView, EnterHpModal


def test_check_hp_validity_bunnelby_family():
    # CP 68 with HP 109 is impossible for both Bunnelby and Diggersby
    is_valid, matched, family_disp = check_hp_validity_for_candidate("Bunnelby / Diggersby", 68, 109)
    assert not is_valid
    assert matched is None
    assert "Bunnelby / Diggersby" in family_disp

    # CP 68 with HP 35 is mathematically possible for Bunnelby
    is_valid, matched, family_disp = check_hp_validity_for_candidate("Bunnelby / Diggersby", 68, 35)
    assert is_valid
    assert matched == "Bunnelby"
    assert "Bunnelby / Diggersby" in family_disp


def test_check_hp_validity_single_species():
    # Pikachu CP 10, HP 10 is valid
    is_valid, matched, family_disp = check_hp_validity_for_candidate("Pikachu", 10, 10)
    assert is_valid
    assert matched == "Pikachu"


def test_impossible_stat_classification_preserves_cp():
    img = cv2.imread("/tmp/bunnelby_prior.png")
    if img is None:
        pytest.skip("Test image not available in /tmp")

    res = classify_pokemon_from_image(img, known_cp=68)

    # Must NOT mutate CP 68 to 580
    assert res["cp"] == 68
    # Must flag as IMPOSSIBLE
    assert res["stat_status"] == "IMPOSSIBLE"
    # Must format family with slashes
    assert res["family_display"] == "Bunnelby / Diggersby"
    assert "cannot have CP 68 with HP 109" in res["stat_error_reason"]


@pytest.mark.asyncio
async def test_enter_hp_modal_valid_submission():
    from unittest.mock import AsyncMock, MagicMock
    from src.bot import get_game_for_channel

    game = get_game_for_channel(999991)
    game.current_cp = 67
    game.last_user_id = 111

    target_msg = MagicMock()
    target_msg.remove_reaction = AsyncMock()
    target_msg.add_reaction = AsyncMock()
    target_msg.author.mention = "<@123>"

    reply_msg = MagicMock()
    reply_msg.edit = AsyncMock()

    corr_msg = MagicMock()
    corr_msg.delete = AsyncMock()

    view = StatCorrectionView(
        bot=MagicMock(),
        target_message=target_msg,
        reply_msg=reply_msg,
        channel_id=999991,
        user_id=123,
        extracted_cp=68,
        detected_hp=109,
        family_display="Bunnelby / Diggersby",
        consecutive_warning="",
        previous_cp=67,
        previous_last_user_id=111
    )
    view.correction_msg = corr_msg

    modal = EnterHpModal(view)
    modal.hp_input = MagicMock()
    modal.hp_input.value = "35"  # Valid HP for Bunnelby CP 68

    interaction = MagicMock()
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()

    await modal.on_submit(interaction)

    # Must switch reaction from 🤨 to ✅
    target_msg.remove_reaction.assert_awaited_once_with("🤨", view.bot.user)
    target_msg.add_reaction.assert_awaited_once_with("✅")

    # Original message edited to approved count
    reply_msg.edit.assert_awaited_once()
    assert "Bunnelby CP 68 ✅" in reply_msg.edit.call_args.kwargs["content"]

    # Other message must disappear
    corr_msg.delete.assert_awaited_once()

    # Game state advanced
    assert game.current_cp == 68
    assert game.last_user_id == 123


@pytest.mark.asyncio
async def test_enter_hp_modal_still_impossible_submission():
    from unittest.mock import AsyncMock, MagicMock
    from src.bot import get_game_for_channel

    game = get_game_for_channel(999992)
    game.current_cp = 67
    game.last_user_id = 111

    target_msg = MagicMock()
    target_msg.remove_reaction = AsyncMock()
    target_msg.add_reaction = AsyncMock()
    target_msg.author.mention = "<@123>"

    reply_msg = MagicMock()
    reply_msg.edit = AsyncMock()

    corr_msg = MagicMock()
    corr_msg.edit = AsyncMock()
    corr_msg.delete = AsyncMock()

    view = StatCorrectionView(
        bot=MagicMock(),
        target_message=target_msg,
        reply_msg=reply_msg,
        channel_id=999992,
        user_id=123,
        extracted_cp=68,
        detected_hp=109,
        family_display="Bunnelby / Diggersby",
        consecutive_warning="",
        previous_cp=67,
        previous_last_user_id=111
    )
    view.correction_msg = corr_msg

    modal = EnterHpModal(view)
    modal.hp_input = MagicMock()
    modal.hp_input.value = "110"  # Still impossible

    interaction = MagicMock()
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()

    await modal.on_submit(interaction)

    # Reaction must NOT change
    target_msg.remove_reaction.assert_not_awaited()
    target_msg.add_reaction.assert_not_awaited()

    # Original message must NOT change
    reply_msg.edit.assert_not_awaited()

    # Correction message with button must be deleted
    corr_msg.delete.assert_awaited_once()

    # Ephemeral message must be sent
    interaction.response.send_message.assert_awaited_once()
    assert interaction.response.send_message.call_args.kwargs["ephemeral"] is True
    assert "still mathematically impossible" in interaction.response.send_message.call_args.args[0]

    # Game state must NOT advance
    assert game.current_cp == 67


