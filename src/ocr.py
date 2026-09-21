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


try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def _load_image(image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray]) -> np.ndarray:
    """Loads and converts various input formats into a BGR numpy ndarray, with HEIC support."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            # Fallback to PIL (handles HEIC, WebP, etc.)
            try:
                pil_img = Image.open(image_input)
                rgb = np.array(pil_img.convert("RGB"))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError(f"Could not read image file from path: {image_input}")
        return img

    if isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            try:
                pil_img = Image.open(io.BytesIO(image_input))
                rgb = np.array(pil_img.convert("RGB"))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError("Could not decode image from bytes")
        return img

    if isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        try:
            pil_img = Image.open(image_input)
            rgb = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
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


from src.ocr_engine import run_fast_ocr


def _parse_all_candidates(text: str) -> List[Tuple[bool, int]]:
    """
    Finds all potential CP numbers in OCR text.
    Enforces valid Pokemon GO CP range: 10 <= CP <= 6000.
    Returns a list of tuples: (has_cp_prefix: bool, value: int)
    """
    if not text:
        return []

    results = []
    # Priority 1: Match with explicit CP prefix or common OCR misreads (e.g. "cp", "ce", "cep", "ep", "gp", "p", "c11")
    # Note: Require 'c' to be immediately followed by digits (e.g. 'c11') rather than allowing lone 'c ' (e.g. 'c 172')
    # where letter 'P' in 'CP' was misread as digit '1'.
    for m in re.finditer(r'(?:cp|ce|cep|ep|gp|op|dp|p|c(?=\d))\s*[:\-\s]?\s*([1-9]\d{1,3})\b', text, re.IGNORECASE):
        v = int(m.group(1))
        if 10 <= v <= 6000:
            results.append((True, v))

    # Priority 2: Standalone integer tokens
    for m in re.finditer(r'(?:^|[^\d])([1-9]\d{1,3})(?:[^\d]|$)', text):
        v = int(m.group(1))
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
    Uses ultra-fast in-memory OCR with multi-channel decomposition,
    top-anchored search (for cropped/chopped screens), and frequency voting.
    """
    img = _load_image(image_input)
    h, w = img.shape[:2]

    # Normalize resolution
    max_h = 1080
    if h > max_h:
        scale = max_h / float(h)
        new_w = int(round(w * scale))
        img = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    crops = [
        (int(h * 0.048), int(h * 0.098), int(w * 0.20), int(w * 0.80)),
        (int(h * 0.040), int(h * 0.120), int(w * 0.20), int(w * 0.80)),
        (int(h * 0.012), int(h * 0.085), int(w * 0.18), int(w * 0.82)),
        (int(h * 0.030), int(h * 0.130), int(w * 0.18), int(w * 0.82)),
        (int(h * 0.070), int(h * 0.155), int(w * 0.18), int(w * 0.82))
    ]

    whitelist = "CPcp0123456789 \n"

    # Fast Path (Poké Genie style): tight standard crop with white / blue thresholds
    # Resolves ~90% of standard screenshots in 1-2 calls (<15ms)
    c0 = img[crops[0][0]:crops[0][1], crops[0][2]:crops[0][3]]
    if c0.size > 0:
        min_bgr0 = np.minimum(c0[:, :, 0], np.minimum(c0[:, :, 1], c0[:, :, 2]))
        blue0 = c0[:, :, 0]
        for chan in (min_bgr0, blue0):
            s0 = cv2.resize(chan, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
            for th in (230, 240):
                _, b0 = cv2.threshold(s0, th, 255, cv2.THRESH_BINARY_INV)
                bord0 = cv2.copyMakeBorder(b0, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                t0 = run_fast_ocr(bord0, psm=7, whitelist=whitelist)
                cands0 = [c[1] for c in _parse_all_candidates(t0) if c[0]]
                if cands0:
                    return cands0[0]

    # Pass 1: Prioritize explicit CP-prefixed matches across crops & channels
    all_prefixed: List[int] = []
    for (y1, y2, x1, x2) in crops:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        min_bgr = np.minimum(crop[:, :, 0], np.minimum(crop[:, :, 1], crop[:, :, 2]))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        channels = [crop[:, :, 0], min_bgr, gray]

        crop_prefixed: List[int] = []

        for th in [215, 225, 230, 245]:
            for chan in channels:
                scaled = cv2.resize(chan, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
                _, b = cv2.threshold(scaled, th, 255, cv2.THRESH_BINARY_INV)
                bordered = cv2.copyMakeBorder(b, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                
                txt = run_fast_ocr(bordered, psm=7, whitelist=whitelist)
                found_here = False
                for is_p, v in _parse_all_candidates(txt):
                    if is_p:
                        crop_prefixed.append(v)
                        all_prefixed.append(v)
                        found_here = True
                if not found_here:
                    txt6 = run_fast_ocr(bordered, psm=6, whitelist=whitelist)
                    for is_p, v in _parse_all_candidates(txt6):
                        if is_p:
                            crop_prefixed.append(v)
                            all_prefixed.append(v)

            # Early exit: if we have consistent prefixed candidate agreement, break early
            if len(crop_prefixed) >= 2 and Counter(crop_prefixed).most_common(1)[0][1] >= 2:
                break

        if crop_prefixed:
            counts = Counter(crop_prefixed)
            if counts.most_common(1)[0][1] >= 2:
                valid_cands = [k for k, c in counts.items() if c >= 2]
                return sorted(valid_cands, key=lambda k: (len(str(k)), counts[k]), reverse=True)[0]

    if all_prefixed:
        counts = Counter(all_prefixed)
        return sorted(counts.keys(), key=lambda k: (counts[k], len(str(k))), reverse=True)[0]

    # Pass 2: Fallback for sprites where CP letters are obscured (e.g. Gimmighoul coin rim)
    unprefixed: List[int] = []
    for (y1, y2, x1, x2) in crops[:2]:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        min_bgr = np.minimum(crop[:, :, 0], np.minimum(crop[:, :, 1], crop[:, :, 2]))
        channels = [crop[:, :, 0], min_bgr]

        for th in (230, 220):
            for chan in channels:
                scaled = cv2.resize(chan, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
                _, b = cv2.threshold(scaled, th, 255, cv2.THRESH_BINARY_INV)
                bordered = cv2.copyMakeBorder(b, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                txt = run_fast_ocr(bordered, psm=7, whitelist=whitelist)
                for is_p, v in _parse_all_candidates(txt):
                    if not is_p:
                        unprefixed.append(v)

        if unprefixed and len(unprefixed) >= 2:
            counts = Counter(unprefixed)
            if counts.most_common(1)[0][1] >= 2:
                return counts.most_common(1)[0][0]

    return None



