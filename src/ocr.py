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


def _parse_all_candidates(text: str) -> List[Tuple[bool, int]]:
    """
    Finds all potential CP numbers in OCR text.
    Enforces valid Pokemon GO CP range: 10 <= CP <= 6000.
    Returns a list of tuples: (has_cp_prefix: bool, value: int)
    """
    if not text:
        return []

    results = []
    # Priority 1: Match with explicit CP / ce / cep / ep prefix
    for m in re.finditer(r'(?:cp|ce|cep|ep)\s*[:\-\s]?\s*(\d+)\b', text, re.IGNORECASE):
        v = int(m.group(1))
        if 10 <= v <= 6000:
            results.append((True, v))

    # Priority 2: Standalone integer tokens
    for token in re.findall(r'\b\d+\b', text):
        v = int(token)
        if 10 <= v <= 6000:
            results.append((False, v))

    return results


def _parse_cp_text(text: str) -> Optional[int]:
    """Helper for direct text parsing tests."""
    cands = _parse_all_candidates(text)
    if not cands:
        return None
    prefixed = [c[1] for c in cands if c[0]]
    if prefixed:
        return prefixed[0]
    return cands[0][1]


def extract_cp_from_image(image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray]) -> Optional[int]:
    """
    Extracts the Pokémon CP value from a Pokémon GO screenshot.
    Uses multi-thresholding, dual blue/grayscale channel analysis,
    tiered crop inspection, resolution normalization, and frequency voting.
    """
    img = _load_image(image_input)
    h, w = img.shape[:2]

    # Normalize resolution: high-res screenshots (e.g. 2142x960 from iPhone Retina)
    # create excessively thick stroke widths that bridge small gaps (like balloon strings).
    # Downscaling images larger than 1080p height standardizes stroke thickness for Tesseract.
    max_h = 1080
    if h > max_h:
        scale = max_h / float(h)
        new_w = int(round(w * scale))
        img = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # Crop bounding boxes for the CP banner:
    # 1. Standard banner window (covers standard full-screen and common cropped views)
    # 2. Tight vertical window (cuts off balloon strings or high/low background artifacts)
    # 3. Wider banner window (handles varying status-bar heights and wide layouts)
    crops = [
        (int(h * 0.040), int(h * 0.120), int(w * 0.20), int(w * 0.80)),
        (int(h * 0.048), int(h * 0.098), int(w * 0.20), int(w * 0.80)),
        (int(h * 0.030), int(h * 0.130), int(w * 0.18), int(w * 0.82))
    ]

    whitelist = "CPcp0123456789 \n"

    # Pass 1: Look for explicit CP prefix (e.g. "CP 11", "cp5629") across crops
    for (y1, y2, x1, x2) in crops:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        all_found: List[Tuple[bool, int]] = []
        # Multi-channel decomposition:
        # 1. Blue channel: isolator for yellow/green sprites & gold coin backgrounds
        # 2. Min(B,G,R): isolator for pure white CP text against purple/blue sky and backgrounds (e.g. Zacian)
        # 3. Grayscale: standard luminance
        min_bgr = np.minimum(crop[:, :, 0], np.minimum(crop[:, :, 1], crop[:, :, 2]))
        channels = [crop[:, :, 0], min_bgr, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)]

        for chan in channels:
            scaled = cv2.resize(chan, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
            for th in [245, 240, 235, 230, 220, 210]:
                _, b = cv2.threshold(scaled, th, 255, cv2.THRESH_BINARY_INV)
                bordered = cv2.copyMakeBorder(b, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                for psm in (6, 7):
                    try:
                        txt = pytesseract.image_to_string(bordered, config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}").strip()
                        all_found.extend(_parse_all_candidates(txt))
                    except Exception:
                        pass

        prefixed = [c[1] for c in all_found if c[0]]
        if prefixed:
            # Sort by longest candidate first, then frequency (e.g. 5629 beats 62)
            counts = Counter(prefixed)
            return sorted(counts.keys(), key=lambda k: (len(str(k)), counts[k]), reverse=True)[0]

    # Pass 2: Fallback to standalone integer frequency voting if no explicit CP prefix was read
    for (y1, y2, x1, x2) in crops:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        all_found: List[Tuple[bool, int]] = []
        min_bgr = np.minimum(crop[:, :, 0], np.minimum(crop[:, :, 1], crop[:, :, 2]))
        channels = [crop[:, :, 0], min_bgr, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)]

        for chan in channels:
            scaled = cv2.resize(chan, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
            for th in [245, 240, 235, 230, 220, 210]:
                _, b = cv2.threshold(scaled, th, 255, cv2.THRESH_BINARY_INV)
                bordered = cv2.copyMakeBorder(b, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                for psm in (6, 7):
                    try:
                        txt = pytesseract.image_to_string(bordered, config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}").strip()
                        all_found.extend(_parse_all_candidates(txt))
                    except Exception:
                        pass

        if all_found:
            return Counter([c[1] for c in all_found]).most_common(1)[0][0]

    return None

