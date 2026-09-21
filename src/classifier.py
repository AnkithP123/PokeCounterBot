import os
import re
import math
import json
import logging
from typing import Optional, Dict, Any, List, Union
import io
from PIL import Image

import cv2
import numpy as np
import pytesseract
from difflib import get_close_matches

from src.ocr import extract_cp_from_image, _load_image
from src.ocr_engine import run_fast_ocr

logger = logging.getLogger("PokeClassifier")

DATA_DIR = os.path.dirname(__file__)
CPM_PATH = os.path.join(DATA_DIR, "cpm.json")
FAMILIES_PATH = os.path.join(DATA_DIR, "pogo_families.json")

# Load CPM table and family stats
with open(CPM_PATH, "r", encoding="utf-8") as f:
    CPM_TABLE: List[float] = json.load(f)

with open(FAMILIES_PATH, "r", encoding="utf-8") as f:
    FAMILIES: Dict[str, List[Dict[str, Any]]] = json.load(f)


# Build lookup maps for fast species-to-family and name resolution
ALL_SPECIES: Dict[str, str] = {}
SPECIES_TO_FAMILY: Dict[str, str] = {}
for fam_key, members in FAMILIES.items():
    for m in members:
        sp_name = m["name"]
        ALL_SPECIES[sp_name.lower()] = sp_name
        SPECIES_TO_FAMILY[sp_name.lower()] = fam_key


def is_stat_combination_possible(species: Dict[str, Any], target_cp: int, target_hp: int) -> bool:
    """
    Checks if a given target CP and HP can be mathematically produced by a species
    for any valid level (1 to 50 in 0.5 increments) and IV combination (0 to 15).
    """
    base_atk = species["atk"]
    base_def = species["def"]
    base_sta = species["sta"]

    for cpm in CPM_TABLE:
        # 1. Quick filter on possible Stamina IVs
        possible_iv_sta = [
            iv_s for iv_s in range(16)
            if max(10, int((base_sta + iv_s) * cpm)) == target_hp
        ]
        if not possible_iv_sta:
            continue

        cpm2_10 = (cpm ** 2) / 10.0

        for iv_s in possible_iv_sta:
            sqrt_sta = math.sqrt(base_sta + iv_s)
            for iv_d in range(16):
                sqrt_def = math.sqrt(base_def + iv_d)
                term = sqrt_def * sqrt_sta * cpm2_10

                min_cp = max(10, int(base_atk * term))
                max_cp = max(10, int((base_atk + 15) * term))

                if min_cp <= target_cp <= max_cp:
                    for iv_a in range(16):
                        if max(10, int((base_atk + iv_a) * term)) == target_cp:
                            return True

    return False


def extract_hp_from_image(img: np.ndarray) -> Optional[int]:
    """
    Extracts the Pokémon HP value from the 'XX/XX HP' text under the name or appraisal card.
    Supports international variants (HP, KP, PV).
    """
    h, w = img.shape[:2]
    # Fast path: Poke Genie tight HP crop (y: ~0.49-0.59, x: ~0.22-0.78)
    tight_crop = img[int(h * 0.49):int(h * 0.59), int(w * 0.22):int(w * 0.78)]
    if tight_crop.size > 0:
        gray_tight = cv2.cvtColor(tight_crop, cv2.COLOR_BGR2GRAY)
        for th in (0, 200):
            proc = gray_tight if th == 0 else cv2.threshold(gray_tight, th, 255, cv2.THRESH_BINARY)[1]
            txt = run_fast_ocr(proc, psm=7)
            match = re.search(r'(?:(\d+)\s*/\s*)?(\d+)\s*(?:HP|KP|PV)\b', txt, re.IGNORECASE)
            if match:
                return int(match.group(2))

    # Fallback to wider middle section for non-standard or appraisal screens
    mid_crop = img[int(h * 0.40):int(h * 0.65), :]
    gray = cv2.cvtColor(mid_crop, cv2.COLOR_BGR2GRAY)
    for th in (0, 200, 180):
        proc = gray if th == 0 else cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1]
        txt = run_fast_ocr(proc, psm=3)
        match = re.search(r'(?:(\d+)\s*/\s*)?(\d+)\s*(?:HP|KP|PV)\b', txt, re.IGNORECASE)
        if match:
            return int(match.group(2))

    return None


POWERUP_COST_TABLE: Dict[int, Dict[str, Any]] = {
    200: {"candy": 1, "level": "1.0 - 2.5"},
    400: {"candy": 1, "level": "3.0 - 4.5"},
    600: {"candy": 1, "level": "5.0 - 6.5"},
    800: {"candy": 1, "level": "7.0 - 8.5"},
    1000: {"candy": 2, "level": "9.0 - 10.5"},
    1300: {"candy": 2, "level": "11.0 - 12.5"},
    1600: {"candy": 2, "level": "13.0 - 14.5"},
    1900: {"candy": 2, "level": "15.0 - 16.5"},
    2200: {"candy": 2, "level": "17.0 - 18.5"},
    2500: {"candy": 3, "level": "19.0 - 20.5"},
    3000: {"candy": 3, "level": "21.0 - 22.5"},
    3500: {"candy": 3, "level": "23.0 - 24.5"},
    4000: {"candy": 4, "level": "25.0 - 26.5"},
    4500: {"candy": 4, "level": "27.0 - 28.5"},
    5000: {"candy": 4, "level": "29.0 - 30.5"},
}


def validate_or_correct_cp_for_species(species_name: str, detected_cp: Optional[int], hp: Optional[int]) -> Optional[int]:
    """
    Validates detected CP against Pokémon GO's level CPM curve and base stats.
    If detected_cp is impossible (e.g. 20 for Seviper with 17 HP, or 172 for Porygon with 28 HP),
    tests common OCR confusions ('0' <-> '8', '3' <-> '2', '1' <-> '7') and leading digit
    drops (when letter 'P' in 'CP' was misread as a digit, e.g. '173' -> '73') to find
    a mathematically valid CP.
    """
    if not detected_cp or not hp or not species_name:
        return detected_cp

    species_data = None
    fam_key = SPECIES_TO_FAMILY.get(species_name.lower())
    if fam_key and fam_key in FAMILIES:
        for m in FAMILIES[fam_key]:
            if m["name"].lower() == species_name.lower():
                species_data = m
                break

    if not species_data:
        return detected_cp

    # Check if currently detected CP is possible
    if is_stat_combination_possible(species_data, detected_cp, hp):
        return detected_cp

    # If impossible, test OCR digit confusion corrections
    cp_str = str(detected_cp)
    confusions = {
        '0': ['8', '9'],
        '8': ['0', '3'],
        '3': ['2', '8'],
        '2': ['3', '7'],
        '1': ['7', '4'],
        '7': ['1', '2'],
        '6': ['8', '5'],
        '5': ['6']
    }

    candidates_to_test: List[int] = []

    # 1. Single digit substitutions on original string
    for i, ch in enumerate(cp_str):
        if ch in confusions:
            for repl in confusions[ch]:
                c_val = int(cp_str[:i] + repl + cp_str[i+1:])
                if 10 <= c_val <= 6000:
                    candidates_to_test.append(c_val)

    # 2. Leading digit drop (e.g. letter 'P' or border misread as '1', '7', '4' in front of 2-digit CP)
    if len(cp_str) >= 3 and cp_str[0] in ('1', '7', '4'):
        truncated = cp_str[1:]
        t_val = int(truncated)
        if 10 <= t_val <= 6000:
            candidates_to_test.append(t_val)
        for i, ch in enumerate(truncated):
            if ch in confusions:
                for repl in confusions[ch]:
                    c_val = int(truncated[:i] + repl + truncated[i+1:])
                    if 10 <= c_val <= 6000:
                        candidates_to_test.append(c_val)

    for cand_val in candidates_to_test:
        if is_stat_combination_possible(species_data, cand_val, hp):
            logger.info("Auto-corrected impossible CP %d to mathematically verified CP %d for %s (HP %d)",
                        detected_cp, cand_val, species_name, hp)
            return cand_val

    return detected_cp


def extract_card_details_from_image(img: np.ndarray) -> Dict[str, Any]:
    """
    Extracts Total Stardust, Total Candy Count, and Power Up Cost (Stardust & Candy).
    Also estimates the Pokémon Level from the Power Up dust cost.
    """
    h, w = img.shape[:2]

    # Crop the stat counters and power up region (y: 0.50 to 0.85)
    crop = img[int(h * 0.50):int(h * 0.85), :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    txt = run_fast_ocr(gray, psm=6)

    stardust_total = None
    m_dust = re.search(r'(\b\d{1,3}(?:,\d{3})+\b)', txt)
    if m_dust:
        try:
            stardust_total = int(m_dust.group(1).replace(',', ''))
        except ValueError:
            pass
    if not stardust_total:
        m_dust_plain = re.search(r'\b(\d{5,8})\b', txt)
        if m_dust_plain:
            try:
                stardust_total = int(m_dust_plain.group(1))
            except ValueError:
                pass

    candy_total = None
    m_candy = re.search(r'(?:,\d{3}|\d{5,8})\s*[@#oO\(\[\{]?\s*([0-9]{1,4})\b', txt)
    if m_candy:
        try:
            candy_total = int(m_candy.group(1))
        except ValueError:
            pass

    powerup_dust = None
    powerup_candy = None
    est_level = None

    for line in txt.splitlines():
        nums = [int(n) for n in re.findall(r'\b\d+\b', line)]
        for idx, n in enumerate(nums):
            if n in POWERUP_COST_TABLE:
                powerup_dust = n
                table_info = POWERUP_COST_TABLE[n]
                powerup_candy = table_info["candy"]
                est_level = table_info["level"]
                break
        if powerup_dust:
            break

    if not powerup_dust:
        m_pu = re.search(r'(?:POWER\s*UP|[}><\|])\s*\{?\s*(\d{3,4})\s*[^0-9\n]?\s*([1-9])\b', txt, re.IGNORECASE)
        if m_pu:
            dust_val = int(m_pu.group(1))
            if dust_val in POWERUP_COST_TABLE:
                powerup_dust = dust_val
                table_info = POWERUP_COST_TABLE[dust_val]
                powerup_candy = table_info["candy"]
                est_level = table_info["level"]

    return {
        "stardust": stardust_total,
        "candy_count": candy_total,
        "powerup_stardust": powerup_dust,
        "powerup_candy": powerup_candy,
        "estimated_level": est_level
    }


_VIT_MODEL = None
_VIT_PROCESSOR = None


def _get_vit_classifier():
    global _VIT_MODEL, _VIT_PROCESSOR
    if _VIT_MODEL is None:
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
            model_id = "skshmjn/Pokemon-classifier-gen9-1025"
            _VIT_PROCESSOR = ViTImageProcessor.from_pretrained(model_id, local_files_only=True)
            _VIT_MODEL = ViTForImageClassification.from_pretrained(model_id, local_files_only=True)
            _VIT_MODEL.eval()
        except Exception as e:
            logger.warning("Could not load local ViT classifier: %s", e)
            return None, None
    return _VIT_MODEL, _VIT_PROCESSOR


def disambiguate_candidates_with_vision(img: np.ndarray, candidates: List[str]) -> Optional[str]:
    """
    Uses the Vision Transformer (ViT) to rank multiple candidate species from the sprite image.
    Returns the candidate species with the highest prediction logit/probability.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    model, processor = _get_vit_classifier()
    if model is None or processor is None:
        return candidates[0]

    try:
        import torch
        # Crop the sprite area in the upper-center of the screen
        h, w = img.shape[:2]
        sprite_crop = img[int(h * 0.12):int(h * 0.46), int(w * 0.15):int(w * 0.85)]
        rgb = cv2.cvtColor(sprite_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]

        label_to_id = {v.lower(): k for k, v in model.config.id2label.items()}
        scored_candidates = []
        for cand in candidates:
            cand_id = label_to_id.get(cand.lower())
            if cand_id is not None:
                score = logits[cand_id].item()
                scored_candidates.append((score, cand))
            else:
                scored_candidates.append((-999.0, cand))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1]
    except Exception as ex:
        logger.warning("Error running ViT disambiguation: %s", ex)
        return candidates[0]


def extract_candy_name_from_image(img: np.ndarray) -> Optional[str]:
    """
    Extracts the Pokémon evolutionary family name from the 'XXXX CANDY' line.
    Supports English, French, German, Spanish, Italian, and Portuguese variants.
    """
    h, w = img.shape[:2]
    ignored_words = {
        "STARDUST", "POWER", "POW", "GYMS", "RAIDS", "TRAINER", "BATTLES",
        "THE", "AND", "FOR", "NEW", "ATTACK", "EVOLVE", "WEIGHT", "HEIGHT"
    }

    # Fast path: Poke Genie tight Candy crop (y: ~0.70-0.84, x: ~0.45-0.98)
    tight_crop = img[int(h * 0.70):int(h * 0.84), int(w * 0.45):int(w * 0.98)]
    if tight_crop.size > 0:
        gray_tight = cv2.cvtColor(tight_crop, cv2.COLOR_BGR2GRAY)
        for th in (0, 210):
            proc = gray_tight if th == 0 else cv2.threshold(gray_tight, th, 255, cv2.THRESH_BINARY)[1]
            txt = run_fast_ocr(proc, psm=6)
            matches = re.findall(
                r'([A-Za-z]{3,})\s*(?:CANDY|GANDY|CANDV|BONBONS?|CARAMEL(?:LE|OS)?|DOCES)\b',
                txt,
                re.IGNORECASE
            )
            for cand in matches:
                w_u = cand.upper()
                if w_u not in ignored_words:
                    closest = get_close_matches(w_u, FAMILIES.keys(), n=1, cutoff=0.7)
                    if closest:
                        return closest[0]

    # Fallback to wider region if tight crop misses (e.g. non-standard aspect ratios)
    lower_crop = img[int(h * 0.55):int(h * 0.85), :]
    gray = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)
    for th in (210, 0):
        proc = gray if th == 0 else cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1]
        scaled = cv2.resize(proc, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        txt = run_fast_ocr(scaled, psm=6)
        matches = re.findall(
            r'([A-Za-z]{3,})\s*(?:CANDY|GANDY|CANDV|BONBONS?|CARAMEL(?:LE|OS)?|DOCES)\b',
            txt,
            re.IGNORECASE
        )
        for cand in matches:
            w_u = cand.upper()
            if w_u not in ignored_words:
                closest = get_close_matches(w_u, FAMILIES.keys(), n=1, cutoff=0.7)
                if closest:
                    return closest[0]

    return None


def extract_caught_header_species(img: np.ndarray) -> Optional[str]:
    """
    Extracts the official species from the game's client caught stamp:
    e.g. 'This Mewtwo was caught on 7/11/2026', 'This Nidoking was caught on...'
    This ignores any custom nicknames given to the Pokémon.
    """
    h, w = img.shape[:2]
    # Fast path: tight bottom caught card banner
    tight_caught = img[int(h * 0.75):int(h * 0.96), int(w * 0.05):int(w * 0.95)]
    if tight_caught.size > 0:
        gray_tight = cv2.cvtColor(tight_caught, cv2.COLOR_BGR2GRAY)
        for th in (0, 200):
            proc = gray_tight if th == 0 else cv2.threshold(gray_tight, th, 255, cv2.THRESH_BINARY)[1]
            txt = run_fast_ocr(proc, psm=6)
            match = re.search(r'This\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+was\s+caught', txt, re.IGNORECASE)
            if match:
                raw_name = match.group(1).strip().lower()
                if raw_name in ALL_SPECIES:
                    return ALL_SPECIES[raw_name]
                closest = get_close_matches(raw_name, ALL_SPECIES.keys(), n=1, cutoff=0.75)
                if closest:
                    return ALL_SPECIES[closest[0]]

    # Fallback to wider lower half
    lower = img[int(h * 0.45):, :]
    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    for th in (0, 200):
        proc = gray if th == 0 else cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1]
        txt = run_fast_ocr(proc, psm=6)
        match = re.search(r'This\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+was\s+caught', txt, re.IGNORECASE)
        if match:
            raw_name = match.group(1).strip().lower()
            if raw_name in ALL_SPECIES:
                return ALL_SPECIES[raw_name]
            closest = get_close_matches(raw_name, ALL_SPECIES.keys(), n=1, cutoff=0.75)
            if closest:
                return ALL_SPECIES[closest[0]]

    return None


def detect_appraisal_screen(img: np.ndarray) -> bool:
    """
    Detects whether the screenshot is an Appraisal screen (Attack/Defense/HP IV bars).
    Fast check: checks for colored orange/red IV bar pixels (<1ms).
    """
    h, w = img.shape[:2]
    bars_crop = img[int(h * 0.65):int(h * 0.95), int(w * 0.15):int(w * 0.90)]
    if bars_crop.size > 0:
        bgr_orange = np.array([25, 148, 232], dtype=float)
        bgr_red = np.array([121, 127, 228], dtype=float)
        diff_o = np.linalg.norm(bars_crop.astype(float) - bgr_orange, axis=2)
        diff_r = np.linalg.norm(bars_crop.astype(float) - bgr_red, axis=2)
        if np.sum((diff_o < 60) | (diff_r < 60)) >= 200:
            return True

    bottom_crop = img[int(h * 0.50):, :]
    gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)
    txt = run_fast_ocr(gray, psm=3)
    appraisal_keywords = {"attack", "defense", "appraise", "appraisal"}
    found_keywords = sum(1 for kw in appraisal_keywords if re.search(r'\b' + kw + r'\b', txt, re.IGNORECASE))
    return found_keywords >= 1 or "LUCKY POKEMON" in txt.upper()


def extract_appraisal_species(img: np.ndarray) -> Optional[str]:
    """
    Extracts the species name from the appraisal screen header.
    """
    h, w = img.shape[:2]
    # Appraisal card name is in upper area of bottom sheet: y ~ 0.42 to 0.60
    card_top = img[int(h * 0.42):int(h * 0.60), :]
    gray = cv2.cvtColor(card_top, cv2.COLOR_BGR2GRAY)

    for th in (0, 200, 180):
        proc = gray if th == 0 else cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1]
        txt = run_fast_ocr(proc, psm=6)
        for line in txt.split("\n"):
            cleaned = re.sub(r'[^A-Za-z]', '', line.strip()).lower()
            if cleaned in ALL_SPECIES and len(cleaned) >= 3:
                return ALL_SPECIES[cleaned]
            closest = get_close_matches(cleaned, ALL_SPECIES.keys(), n=1, cutoff=0.82)
            if closest and len(cleaned) >= 4:
                return ALL_SPECIES[closest[0]]

    return None


def extract_primary_name_from_image(img: np.ndarray) -> Optional[str]:
    """
    Extracts the large Pokémon species name / nickname displayed under the sprite.
    """
    h, w = img.shape[:2]
    name_crop = img[int(h * 0.37):int(h * 0.53), :]
    gray = cv2.cvtColor(name_crop, cv2.COLOR_BGR2GRAY)

    for th in (0, 220, 200, 180):
        proc = gray if th == 0 else cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1]
        txt = run_fast_ocr(proc, psm=6)
        for line in txt.split("\n"):
            # Strip common nickname tags (symbols, gender markers, exclamation marks)
            cleaned = re.sub(r'[^A-Za-z]', '', line.strip()).lower()
            if cleaned in ALL_SPECIES and len(cleaned) >= 3:
                return ALL_SPECIES[cleaned]
            closest = get_close_matches(cleaned, ALL_SPECIES.keys(), n=1, cutoff=0.85)
            if closest and len(cleaned) >= 4:
                return ALL_SPECIES[closest[0]]

    return None


def extract_appraisal_ivs(img: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Measures the 3 appraisal IV bars (Attack, Defense, Stamina) from 0 to 15.
    Uses pixel column thresholding inspired by Poké Genie's checkIVBarPercentage.
    """
    h, w = img.shape[:2]
    # IV bars reside in bottom-right quarter of appraisal card
    bars_crop = img[int(h * 0.65):int(h * 0.95), int(w * 0.15):int(w * 0.90)]
    if bars_crop.size == 0:
        return None

    # Target colors (BGR):
    # Orange: (25, 148, 232)
    # Max red/pink: (121, 127, 228)
    bgr_orange = np.array([25, 148, 232], dtype=float)
    bgr_red = np.array([121, 127, 228], dtype=float)

    diff_o = np.linalg.norm(bars_crop.astype(float) - bgr_orange, axis=2)
    diff_r = np.linalg.norm(bars_crop.astype(float) - bgr_red, axis=2)
    mask = (diff_o < 60) | (diff_r < 60)

    # If insufficient colored bar pixels found, not an active appraisal
    if np.sum(mask) < 200:
        return None

    # Find vertical profile of the 3 distinct bars
    row_counts = np.sum(mask, axis=1)
    bar_rows = np.where(row_counts > 20)[0]
    if len(bar_rows) == 0:
        return None

    # Group rows into 3 bar segments
    diffs = np.diff(bar_rows)
    split_points = np.where(diffs > 10)[0]
    segments = []
    prev = 0
    for sp in split_points:
        segments.append(bar_rows[prev:sp + 1])
        prev = sp + 1
    segments.append(bar_rows[prev:])

    if len(segments) != 3:
        return None

    iv_vals = []
    # Maximum bar width reference
    col_counts_all = np.sum(mask, axis=0)
    bar_cols = np.where(col_counts_all > 5)[0]
    if len(bar_cols) == 0:
        return None
    max_w = float(bar_cols.max() - bar_cols.min())
    if max_w <= 0:
        return None

    for seg in segments:
        seg_mask = mask[seg, :]
        seg_cols = np.where(np.sum(seg_mask, axis=0) > 0)[0]
        if len(seg_cols) == 0:
            iv_vals.append(0)
            continue
        fill_w = float(seg_cols.max() - seg_cols.min() + 1)
        ratio = min(1.0, fill_w / max_w)
        iv_stat = int(round(ratio * 15.0))
        iv_vals.append(max(0, min(15, iv_stat)))

    atk, dfn, sta = iv_vals[0], iv_vals[1], iv_vals[2]
    percent = round(((atk + dfn + sta) / 45.0) * 100.0, 1)

    return {
        "atk": atk,
        "def": dfn,
        "sta": sta,
        "percent": percent
    }


_UNKNOWN = object()


def _classify_pokemon_core(img: np.ndarray, cp: Optional[int]) -> Dict[str, Any]:
    """Internal classifier core for species and CP/HP triangulation."""
    result: Dict[str, Any] = {
        "species": None,
        "cp": cp,
        "hp": None,
        "candy_family": None,
        "candidates": [],
        "explanation": "",
        "appraisal_ivs": None
    }

    # Detect Appraisal IV bars (Attack/Defense/Stamina) if active screen (<1ms check)
    if detect_appraisal_screen(img):
        result["appraisal_ivs"] = extract_appraisal_ivs(img)

    # Signal 1: Check Candy Family (fastest and covers ~90% of box screenshots)
    candy = extract_candy_name_from_image(img)
    if candy:
        result["candy_family"] = candy
        family_members = FAMILIES.get(candy, [])
        names = [m["name"] for m in family_members]

        hp = extract_hp_from_image(img)
        result["hp"] = hp

        if cp is None or hp is None:
            if len(family_members) == 1:
                result["species"] = family_members[0]["name"]
                result["candidates"] = [family_members[0]["name"]]
                result["explanation"] = f"Identified as {result['species']} (single-stage family with {candy} Candy)."
                return result
            primary_name = extract_primary_name_from_image(img)
            if primary_name and primary_name in names:
                result["species"] = primary_name
                result["candidates"] = names
                result["explanation"] = f"Identified as {primary_name} from name OCR ({candy} Candy family)."
                return result
            caught_species = extract_caught_header_species(img)
            if caught_species and caught_species in names:
                result["species"] = caught_species
                result["candidates"] = [caught_species]
                result["explanation"] = f"Identified as {caught_species} from official caught text ({candy} Candy family)."
                return result
            winner = disambiguate_candidates_with_vision(img, names)
            result["species"] = winner
            result["candidates"] = names
            result["explanation"] = f"Found {candy} Candy, missing CP/HP; identified as {winner} via visual classifier."
            return result

        # Triangulate using CP and HP
        valid_species = [
            m["name"] for m in family_members
            if is_stat_combination_possible(m, cp, hp)
        ]

        # If triangulation produced 0 matches, check if CP had an OCR confusion (e.g. '0' <-> '8', 20 vs 28)
        if not valid_species and cp and hp:
            for m in family_members:
                corrected = validate_or_correct_cp_for_species(m["name"], cp, hp)
                if corrected and corrected != cp:
                    logger.info("Stat-validated CP: corrected %d to %d for %s", cp, corrected, m["name"])
                    cp = corrected
                    result["cp"] = cp
                    valid_species = [
                        fam["name"] for fam in family_members
                        if is_stat_combination_possible(fam, cp, hp)
                    ]
                    break

        result["candidates"] = valid_species

        if len(valid_species) == 1:
            result["species"] = valid_species[0]
            result["explanation"] = f"Triangulated from {candy} Candy + CP {cp} + HP {hp}."
            return result

        if len(valid_species) > 1:
            primary_name = extract_primary_name_from_image(img)
            if primary_name and primary_name in valid_species:
                result["species"] = primary_name
                result["explanation"] = f"Triangulated {', '.join(valid_species)}; confirmed as {primary_name} from name OCR."
                return result
            caught_species = extract_caught_header_species(img)
            if caught_species and caught_species in valid_species:
                result["species"] = caught_species
                result["explanation"] = f"Triangulated {', '.join(valid_species)}; confirmed as {caught_species} from caught text."
                return result
            winner = disambiguate_candidates_with_vision(img, valid_species)
            result["species"] = winner
            result["explanation"] = f"Triangulated {', '.join(valid_species)} for CP {cp}/HP {hp}; selected {winner} using visual classifier."
            return result

        # Stat triangulation produced 0 matches (e.g. mega or costume): fallback to name, caught or vision
        primary_name = extract_primary_name_from_image(img)
        if primary_name and primary_name in names:
            result["species"] = primary_name
            result["explanation"] = f"No standard stat match in {candy} family; identified as {primary_name} from name OCR."
            return result
        caught_species = extract_caught_header_species(img)
        if caught_species and caught_species in names:
            result["species"] = caught_species
            result["explanation"] = f"No standard stat match in {candy} family; identified as {caught_species} from caught text."
            return result
        winner = disambiguate_candidates_with_vision(img, names)
        result["species"] = winner
        result["explanation"] = f"No standard stat match in {candy} family for CP {cp}/HP {hp}; selected {winner} via visual classifier."
        return result

    # Signal 2: No Candy line found: check caught text (e.g. scrolled screens)
    caught_species = extract_caught_header_species(img)
    if caught_species:
        result["species"] = caught_species
        result["candidates"] = [caught_species]
        result["candy_family"] = SPECIES_TO_FAMILY.get(caught_species.lower())
        result["hp"] = extract_hp_from_image(img)
        result["explanation"] = f"Identified as {caught_species} from official caught text."
        return result

    # Signal 3: Appraisal screen
    if detect_appraisal_screen(img):
        appraisal_species = extract_appraisal_species(img)
        if appraisal_species:
            result["species"] = appraisal_species
            result["candidates"] = [appraisal_species]
            result["candy_family"] = SPECIES_TO_FAMILY.get(appraisal_species.lower())
            result["appraisal_ivs"] = extract_appraisal_ivs(img)
            result["hp"] = extract_hp_from_image(img)
            result["explanation"] = f"Identified as {appraisal_species} from appraisal card header."
            return result

    # Signal 4: Primary name under avatar
    primary_name = extract_primary_name_from_image(img)
    if primary_name:
        result["species"] = primary_name
        result["candidates"] = [primary_name]
        result["candy_family"] = SPECIES_TO_FAMILY.get(primary_name.lower())
        result["hp"] = extract_hp_from_image(img)
        result["explanation"] = f"Identified as {primary_name} from species name OCR."
        return result

    # Signal 5: Full-text fallback check for species names
    try:
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_f, w_f = gray_full.shape[:2]
        if h_f > 800:
            scale_f = 800.0 / float(h_f)
            gray_full = cv2.resize(gray_full, (int(round(w_f * scale_f)), 800), interpolation=cv2.INTER_AREA)
        txt_full = run_fast_ocr(gray_full, psm=3)
        ignored_generic = {"normal", "steel", "fairy", "attack", "defense", "weight", "height", "stardust", "candy", "lucky"}
        for line in txt_full.split("\n"):
            cleaned = re.sub(r'[^A-Za-z]', '', line.strip()).lower()
            if cleaned in ALL_SPECIES and len(cleaned) >= 4 and cleaned not in ignored_generic:
                sp = ALL_SPECIES[cleaned]
                result["species"] = sp
                result["candidates"] = [sp]
                result["candy_family"] = SPECIES_TO_FAMILY.get(cleaned)
                result["hp"] = extract_hp_from_image(img)
                result["explanation"] = f"Identified as {sp} from screen text analysis."
                return result
    except Exception:
        pass

    result["explanation"] = "Could not identify Pokémon species from screen text or candy lines."
    return result


def classify_pokemon_from_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    known_cp: Optional[int] = _UNKNOWN
) -> Dict[str, Any]:
    """
    Identifies Pokémon species from a Pokémon GO screenshot using multi-signal cascade:
      1. Candy Family Extraction
      2. Stat Triangulation (CP + HP + CPM base stat validation)
      3. Client Caught Stamp (official species header)
      4. Appraisal Screen Header & IV Bar Measurement
      5. Primary Species Name OCR
      6. Visual ViT Candidate Disambiguation

    Also extracts and enriches with card details:
      - Stardust total
      - Candy count
      - Power Up cost (stardust + candy)
      - Estimated Pokémon level
    """
    img = _load_image(image_input)
    cp = extract_cp_from_image(img) if known_cp is _UNKNOWN else known_cp

    result = _classify_pokemon_core(img, cp)
    card_details = extract_card_details_from_image(img)
    result.update(card_details)
    return result


