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
