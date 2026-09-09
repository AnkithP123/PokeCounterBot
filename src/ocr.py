import re
import os
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Optional, Union

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


def _parse_cp_text(text: str) -> Optional[int]:
    """
    Parses CP number from OCR text.
    Handles 'CP 10', 'CP10', 'ce10', 'cp 11', 'cP11', etc.
    """
    if not text:
        return None

    # Priority 1: Match with explicit CP / ce prefix
    # Matches 'CP 10', 'ce10', 'cP11', 'CP-1500', etc.
    m = re.search(r'(?:cp|ce|cep|c|p)\s*[:\-\s]?\s*(\d+)\b', text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 10 <= val <= 6000:
            return val

    # Priority 2: Standalone integer token
    tokens = re.findall(r'\b\d+\b', text)
    for token in tokens:
        val = int(token)
        if 10 <= val <= 6000:
            return val

    return None


def extract_cp_from_image(image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray]) -> Optional[int]:
    """
    Extracts the Pokémon CP value from a Pokémon GO screenshot.
    Returns the integer CP value or None if not detected.
    """
    img = _load_image(image_input)
    h, w = img.shape[:2]

    # Crop bounding box candidates for the CP region:
    # In standard mobile screens (aspect ratios 16:9 to 21:9), CP sits below the status bar:
    # y: ~4.0% to 12.5%, x: ~20% to 80%
    crops = [
        # Standard primary crop
        (int(h * 0.040), int(h * 0.125), int(w * 0.20), int(w * 0.80)),
        # Slightly wider and taller crop for edge cases or tablet screens
        (int(h * 0.030), int(h * 0.150), int(w * 0.15), int(w * 0.85))
    ]

    whitelist = "CPcp0123456789 \n"

    for (y1, y2, x1, x2) in crops:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Scale up 2.5x to improve OCR on small font sizes
        scaled = cv2.resize(crop, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # Generate preprocessing variations:
        variations = []

        # 1. Otsu threshold (inverted: white text becomes black on white)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variations.append(otsu)

        # 2. Fixed bright threshold (useful when background is bright daytime/sunset sky)
        _, th220 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        variations.append(th220)

        # 3. Intermediate threshold
        _, th200 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        variations.append(th200)

        # 4. Adaptive thresholding
        adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4
        )
        variations.append(adapt)

        for processed in variations:
            # Add white border around the image so edge characters aren't cut
            bordered = cv2.copyMakeBorder(
                processed, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

            for psm in (6, 7, 8):
                config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
                try:
                    text = pytesseract.image_to_string(bordered, config=config).strip()
                except Exception:
                    continue

                cp = _parse_cp_text(text)
                if cp is not None:
                    return cp

    return None
