import os
import io
import unittest
from PIL import Image
from src.ocr import extract_cp_from_image, _parse_cp_text

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
IMG_10_PATH = os.path.join(SAMPLES_DIR, "cp10_fletchling.png")
IMG_11_PATH = os.path.join(SAMPLES_DIR, "cp11_froakie.png")
IMG_2691_PATH = os.path.join(SAMPLES_DIR, "cp2691_rayquaza.png")
IMG_1_PATH = os.path.join(SAMPLES_DIR, "cp1_tinkatink.png")


class TestOCR(unittest.TestCase):
    def test_extract_cp_fletchling_10(self):
        """Verify that Fletchling screenshot resolves to CP 10."""
        cp = extract_cp_from_image(IMG_10_PATH)
        self.assertEqual(cp, 10)

    def test_extract_cp_froakie_11(self):
        """Verify that Froakie screenshot resolves to CP 11."""
        cp = extract_cp_from_image(IMG_11_PATH)
        self.assertEqual(cp, 11)

    def test_extract_cp_rayquaza_2691(self):
        """Verify that Rayquaza screenshot resolves to CP 2691."""
        cp = extract_cp_from_image(IMG_2691_PATH)
        self.assertEqual(cp, 2691)

    def test_extract_cp_tinkatink_1(self):
        """Verify that Tinkatink screenshot resolves to CP 1."""
        cp = extract_cp_from_image(IMG_1_PATH)
        self.assertEqual(cp, 1)

    def test_extract_cp_from_bytes(self):
        """Verify that loading image from bytes and BytesIO works identically."""
        with open(IMG_10_PATH, "rb") as f:
            img_bytes = f.read()

        cp_bytes = extract_cp_from_image(img_bytes)
        self.assertEqual(cp_bytes, 10)

        cp_io = extract_cp_from_image(io.BytesIO(img_bytes))
        self.assertEqual(cp_io, 10)

    def test_extract_cp_from_pil(self):
        """Verify that loading image from PIL.Image works."""
        pil_img = Image.open(IMG_11_PATH)
        cp = extract_cp_from_image(pil_img)
        self.assertEqual(cp, 11)

    def test_parse_cp_text(self):
        """Verify text parsing regex against common OCR variations."""
        self.assertEqual(_parse_cp_text("CP 10"), 10)
        self.assertEqual(_parse_cp_text("CP 1"), 1)
        self.assertEqual(_parse_cp_text("cp10"), 10)
        self.assertEqual(_parse_cp_text("ce10"), 10)
        self.assertEqual(_parse_cp_text("cP 1500"), 1500)
        self.assertEqual(_parse_cp_text("CP-250"), 250)
        self.assertIsNone(_parse_cp_text("No CP here"))
        self.assertIsNone(_parse_cp_text("CP 9999999"))


if __name__ == "__main__":
    unittest.main()
