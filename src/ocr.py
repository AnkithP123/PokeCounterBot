import re
import os
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Optional, Union, List, Tuple
from collections import Counter

# Configure custom tesseract path if set in environment
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _load_image(image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray]) -> np.ndarray:
    """Loads and converts various input formats into a BGR numpy ndarray."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image file from path: {image_input}")
        return img

    if isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes")
        return img

    if isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        nparr = np.frombuffer(image_input.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from BytesIO")
        return img

    if isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(image_input, np.ndarray):
        return image_input

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def _parse_all_candidates(text: str) -> List[Tuple[bool, int, int]]:
    """
    Finds all potential CP numbers in OCR text.
    Returns a list of tuples: (has_cp_prefix: bool, value: int, digit_length: int)
    """
    if not text:
        return []

    results = []
    # Priority 1: Match with explicit CP / ce / cp / c / p prefix
    for m in re.finditer(r'(?:cp|ce|cep|c|p)\s*[:\-\s]?\s*(\d+)\b', text, re.IGNORECASE):
        v = int(m.group(1))
        if 10 <= v <= 6000:
            results.append((True, v, len(m.group(1))))

    # Priority 2: Standalone integer tokens
    for token in re.findall(r'\b\d+\b', text):
        v = int(token)
        if 10 <= v <= 6000:
            results.append((False, v, len(token)))

    return results


def _parse_cp_text(text: str) -> Optional[int]:
    """Helper for direct text parsing tests."""
    cands = _parse_all_candidates(text)
    if not cands:
        return None
    # Prioritize prefixed, then longest digit length
    cands.sort(key=lambda x: (x[0], x[2]), reverse=True)
    return cands[0][1]


def extract_cp_from_image(image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray]) -> Optional[int]:
    """
    Extracts the Pokémon CP value from a Pokémon GO screenshot.
    Uses multi-thresholding and candidate scoring to handle complex Pokémon
    models, horns, sparkles, and varying sky backgrounds.
    """
    img = _load_image(image_input)
    h, w = img.shape[:2]

    # Crop bounding box candidates for the CP region:
    # Standard mobile screens: y: ~4.0% to 13.0%, x: ~18% to 82%
    crops = [
        (int(h * 0.040), int(h * 0.130), int(w * 0.18), int(w * 0.82)),
        (int(h * 0.030), int(h * 0.150), int(w * 0.15), int(w * 0.85))
    ]

    whitelist = "CPcp0123456789 \n"

    for (y1, y2, x1, x2) in crops:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        scaled = cv2.resize(crop, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        all_found: List[Tuple[bool, int, int]] = []

        # High thresholds isolate pure white font (>220-245) from colored Pokémon elements (horns, whiskers, clouds)
        thresholds = [240, 235, 230, 220, 210]
        for th in thresholds:
            _, b = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)
            bordered = cv2.copyMakeBorder(b, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            for psm in (6, 7):
                config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
                try:
                    txt = pytesseract.image_to_string(bordered, config=config).strip()
                except Exception:
                    continue
                all_found.extend(_parse_all_candidates(txt))

        # Otsu thresholding as robust adaptive fallback
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bordered = cv2.copyMakeBorder(otsu, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        try:
            txt = pytesseract.image_to_string(bordered, config=f"--psm 6 -c tessedit_char_whitelist={whitelist}").strip()
            all_found.extend(_parse_all_candidates(txt))
        except Exception:
            pass

        if all_found:
            # Score candidates:
            # 1. Prefer candidate with explicit CP prefix
            prefixed = [c for c in all_found if c[0]]
            pool = prefixed if prefixed else all_found

            # 2. Sort by prefix presence, then by digit length descending (e.g. 2691 > 269)
            pool.sort(key=lambda x: (x[0], x[2]), reverse=True)
            max_len = pool[0][2]

            # 3. Take the most frequently detected value among the longest matches
            best_vals = [c[1] for c in pool if c[2] == max_len]
            most_common = Counter(best_vals).most_common(1)[0][0]
            return most_common

    return None
