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

    for th in (0, 200, 180, 220, 160):
        if th > 0:
            _, proc = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        else:
            proc = gray
        txt = pytesseract.image_to_string(proc)
        match = re.search(r'(\d+)\s*/\s*(\d+)\s*HP', txt, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


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
    Candy Family extraction + CP & HP formula triangulation + ViT candidate disambiguation.

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

    names = [m["name"] for m in family_members]

    if not cp or not hp:
        # Fallback to single-member family if unevolved/no evolutions, else vision
        if len(family_members) == 1:
            result["species"] = family_members[0]["name"]
            result["candidates"] = [family_members[0]["name"]]
            result["explanation"] = f"Identified as {result['species']} (single-stage family with {candy} Candy)."
        else:
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
    result["candidates"] = valid_species

    if len(valid_species) == 1:
        result["species"] = valid_species[0]
        result["explanation"] = f"Triangulated from {candy} Candy + CP {cp} + HP {hp}."
    elif len(valid_species) > 1:
        winner = disambiguate_candidates_with_vision(img, valid_species)
        result["species"] = winner
        result["explanation"] = f"Triangulated {', '.join(valid_species)} for CP {cp}/HP {hp}; selected {winner} using visual classifier."
    else:
        # If triangulation produced 0 matches (e.g. mega/costume/special boost), fallback using vision
        winner = disambiguate_candidates_with_vision(img, names)
        result["species"] = winner
        result["explanation"] = f"No standard stat match in {candy} family for CP {cp}/HP {hp}; selected {winner} via visual classifier."

    return result
