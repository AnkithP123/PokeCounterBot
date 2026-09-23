import os
import threading
import logging
from typing import Optional, Union
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Configure custom tesseract path if set in environment
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

_HAS_TESSEROCR = False
_TESSDATA_PATH = None
_local = threading.local()

LOCAL_TESSDATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tessdata")

try:
    import tesserocr

    tessdata_candidates = [
        os.environ.get("TESSDATA_PREFIX", ""),
        "/opt/homebrew/share/tessdata",
        "/usr/local/share/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/5",
        "/usr/share/tessdata",
        "/usr/share",
        LOCAL_TESSDATA,
    ]
    for p in tessdata_candidates:
        if p and os.path.exists(p):
            try:
                with tesserocr.PyTessBaseAPI(path=p) as test_api:
                    _TESSDATA_PATH = p
                    _HAS_TESSEROCR = True
                    logger.info("Using tesserocr with tessdata path: %s", _TESSDATA_PATH)
                    break
            except Exception:
                continue

    if not _HAS_TESSEROCR:
        try:
            with tesserocr.PyTessBaseAPI() as test_api:
                _HAS_TESSEROCR = True
                _TESSDATA_PATH = None
                logger.info("Using tesserocr with default path")
        except Exception:
            _HAS_TESSEROCR = False
except Exception as e:
    logger.warning("tesserocr not available (%s), falling back to pytesseract", e)
    _HAS_TESSEROCR = False

import pytesseract
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _get_thread_api(lang: str = "eng"):
    """Returns a thread-local PyTessBaseAPI instance for zero-overhead in-memory OCR."""
    if not hasattr(_local, "apis"):
        _local.apis = {}

    if lang not in _local.apis:
        try:
            if lang == "number" and os.path.exists(os.path.join(LOCAL_TESSDATA, "number.traineddata")):
                api = tesserocr.PyTessBaseAPI(path=LOCAL_TESSDATA, lang="number")
            elif _TESSDATA_PATH:
                api = tesserocr.PyTessBaseAPI(path=_TESSDATA_PATH, lang=lang)
            else:
                api = tesserocr.PyTessBaseAPI(lang=lang)
            _local.apis[lang] = api
        except Exception:
            # If custom language (e.g. number) is not available, fallback to eng
            if lang != "eng":
                return _get_thread_api("eng")
            raise

    return _local.apis[lang]


def run_fast_ocr(
    image: Union[np.ndarray, Image.Image],
    psm: int = 3,
    whitelist: Optional[str] = None,
    lang: str = "eng"
) -> str:
    """
    Executes OCR on an image (numpy BGR/grayscale or PIL Image).
    Uses in-process C++ TessBaseAPI via tesserocr (~2-5ms per call) if available,
    falling back to pytesseract (~100ms per call) transparently.
    """
    if image is None:
        return ""

    if isinstance(image, np.ndarray):
        if image.size == 0:
            return ""
        pil_img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        return ""

    if _HAS_TESSEROCR:
        try:
            api = _get_thread_api(lang)
            # Set whitelist
            if whitelist:
                api.SetVariable("tessedit_char_whitelist", whitelist)
            else:
                api.SetVariable("tessedit_char_whitelist", "")

            # Set PSM
            api.SetPageSegMode(psm)

            api.SetImage(pil_img)
            return api.GetUTF8Text().strip()
        except Exception as ex:
            logger.debug("tesserocr call failed (%s), falling back to pytesseract", ex)

    # pytesseract fallback
    cfg = f"--psm {psm}"
    if lang == "number" and os.path.exists(os.path.join(LOCAL_TESSDATA, "number.traineddata")):
        cfg += f' --tessdata-dir "{LOCAL_TESSDATA}"'
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    try:
        return pytesseract.image_to_string(pil_img, lang=lang, config=cfg).strip()
    except Exception as ex:
        logger.debug("pytesseract call failed: %s", ex)
        return ""
