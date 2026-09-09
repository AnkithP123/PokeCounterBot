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

logger = logging.getLogger("PokeClassifier")

DATA_DIR = os.path.dirname(__file__)
CPM_PATH = os.path.join(DATA_DIR, "cpm.json")
FAMILIES_PATH = os.path.join(DATA_DIR, "pogo_families.json")

# Load CPM table and family stats
with open(CPM_PATH, "r", encoding="utf-8") as f:
    CPM_TABLE: List[float] = json.load(f)

with open(FAMILIES_PATH, "r", encoding="utf-8") as f:
    FAMILIES: Dict[str, List[Dict[str, Any]]] = json.load(f)


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
    Extracts the Pokémon HP value from the 'XX/XX HP' text under the name.
    """
    h, w = img.shape[:2]
    mid_crop = img[int(h * 0.44):int(h * 0.60), :]
    gray = cv2.cvtColor(mid_crop, cv2.COLOR_BGR2GRAY)
    txt = pytesseract.image_to_string(gray)

    match = re.search(r'(\d+)\s*/\s*(\d+)\s*HP', txt, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def extract_candy_name_from_image(img: np.ndarray) -> Optional[str]:
    """
    Extracts the Pokémon evolutionary family name from the 'XXXX CANDY' line.
    """
    h, w = img.shape[:2]
    lower_crop = img[int(h * 0.55):int(h * 0.82), :]
    gray = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)

    ignored_words = {
        "STARDUST", "POWER", "POW", "GYMS", "RAIDS", "TRAINER", "BATTLES",
        "THE", "AND", "FOR", "NEW", "ATTACK", "EVOLVE"
    }

    # Test several binarization thresholds to capture light gray text
    for th in (220, 200, 180, 0):
        if th > 0:
            _, proc = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        else:
            proc = gray

        scaled = cv2.resize(proc, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        txt = pytesseract.image_to_string(scaled, config="--psm 6")

        matches = re.findall(r'([A-Za-z]{3,})\s*(?:CANDY|GANDY|CANDV)\b', txt, re.IGNORECASE)
        for cand in matches:
            w_u = cand.upper()
            if w_u not in ignored_words:
                # Match against known family keys
                closest = get_close_matches(w_u, FAMILIES.keys(), n=1, cutoff=0.7)
                if closest:
                    return closest[0]

    return None


def classify_pokemon_from_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray]
) -> Dict[str, Any]:
    """
    Identifies the Pokémon species from a Pokémon GO screenshot using
    Candy Family extraction + CP & HP formula triangulation.

    Returns a dict with:
      - species: Name of the identified Pokémon (e.g. 'Mightyena') or None
      - cp: Extracted CP value
      - hp: Extracted HP value
      - candy_family: Extracted Candy name (e.g. 'POOCHYENA')
      - candidates: List of possible matching species in the family
      - explanation: Human-readable explanation of how the result was determined
    """
    img = _load_image(image_input)

    cp = extract_cp_from_image(img)
    hp = extract_hp_from_image(img)
    candy = extract_candy_name_from_image(img)

    result: Dict[str, Any] = {
        "species": None,
        "cp": cp,
        "hp": hp,
        "candy_family": candy,
        "candidates": [],
        "explanation": ""
    }

    if not candy:
        result["explanation"] = "Could not find a 'CANDY' line on screen (might be an appraisal or cropped screenshot)."
        return result

    family_members = FAMILIES.get(candy, [])
    if not family_members:
        result["explanation"] = f"Candy family '{candy}' not found in database."
        return result

    if not cp or not hp:
        # Fallback to single-member family if unevolved/no evolutions
        if len(family_members) == 1:
            result["species"] = family_members[0]["name"]
            result["candidates"] = [family_members[0]["name"]]
            result["explanation"] = f"Identified as {result['species']} (single-stage family with {candy} Candy)."
        else:
            names = [m["name"] for m in family_members]
            result["candidates"] = names
            result["explanation"] = f"Found {candy} Candy, but missing CP/HP to differentiate between: {', '.join(names)}."
        return result

    # Triangulate using CP and HP
    valid_species = [
        m["name"] for m in family_members
        if is_stat_combination_possible(m, cp, hp)
    ]
    result["candidates"] = valid_species

    if len(valid_species) == 1:
        result["species"] = valid_species[0]
        result["explanation"] = f"Triangulated from {candy} Candy + CP {cp} + HP {hp}."
    elif len(valid_species) > 1:
        result["species"] = valid_species[0]
        result["explanation"] = f"Ambiguous between {', '.join(valid_species)} for CP {cp} and HP {hp}."
    else:
        # If triangulation produced 0 matches (e.g. mega/costume/special boost), fallback to base form
        result["species"] = family_members[0]["name"]
        result["explanation"] = f"No standard stat match in {candy} family for CP {cp}/HP {hp}; defaulted to {result['species']}."

    return result
