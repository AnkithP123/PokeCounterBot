import os
import io
import unittest
from PIL import Image
from src.ocr import extract_cp_from_image, _parse_cp_text

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
IMG_10_PATH = os.path.join(SAMPLES_DIR, "cp10_fletchling.png")
IMG_11_PATH = os.path.join(SAMPLES_DIR, "cp11_froakie.png")
IMG_2691_PATH = os.path.join(SAMPLES_DIR, "cp2691_rayquaza.png")
IMG_DITTO_PATH = os.path.join(SAMPLES_DIR, "cp10_ditto_background.png")
IMG_42_PATH = os.path.join(SAMPLES_DIR, "cp42_gimmighoul.png")
IMG_475_PATH = os.path.join(SAMPLES_DIR, "cp475_gimmighoul.png")
IMG_PIKACHU_PATH = os.path.join(SAMPLES_DIR, "cp11_flying_pikachu.png")


class TestOCR(unittest.TestCase):
    def test_extract_cp_fletchling_10(self):
        """Verify that Fletchling screenshot resolves to CP 10."""
        self.assertEqual(extract_cp_from_image(IMG_10_PATH), 10)

    def test_extract_cp_froakie_11(self):
        """Verify that Froakie screenshot resolves to CP 11."""
        self.assertEqual(extract_cp_from_image(IMG_11_PATH), 11)

    def test_extract_cp_rayquaza_2691(self):
        """Verify that Rayquaza screenshot resolves to CP 2691."""
        self.assertEqual(extract_cp_from_image(IMG_2691_PATH), 2691)

    def test_extract_cp_ditto_10(self):
        """Verify that Ditto with event background resolves to CP 10."""
        self.assertEqual(extract_cp_from_image(IMG_DITTO_PATH), 10)

    def test_extract_cp_gimmighoul_42(self):
        """Verify that Gimmighoul with gold coin background resolves to CP 42."""
        self.assertEqual(extract_cp_from_image(IMG_42_PATH), 42)

    def test_extract_cp_gimmighoul_475(self):
        """Verify that Gimmighoul with gold coin background resolves to CP 475."""
        self.assertEqual(extract_cp_from_image(IMG_475_PATH), 475)

    def test_extract_cp_flying_pikachu_11(self):
        """Verify that 5th Anniversary Flying Pikachu with balloon strings resolves to CP 11."""
        self.assertEqual(extract_cp_from_image(IMG_PIKACHU_PATH), 11)


    def test_extract_cp_from_bytes(self):
        """Verify that loading image from bytes and BytesIO works identically."""
        with open(IMG_10_PATH, "rb") as f:
            img_bytes = f.read()

        self.assertEqual(extract_cp_from_image(img_bytes), 10)
        self.assertEqual(extract_cp_from_image(io.BytesIO(img_bytes)), 10)

    def test_extract_cp_from_pil(self):
        """Verify that loading image from PIL.Image works."""
        pil_img = Image.open(IMG_11_PATH)
        self.assertEqual(extract_cp_from_image(pil_img), 11)

    def test_parse_cp_text(self):
        """Verify text parsing regex against common OCR variations."""
        self.assertEqual(_parse_cp_text("CP 10"), 10)
        self.assertEqual(_parse_cp_text("cp10"), 10)
        self.assertEqual(_parse_cp_text("ce10"), 10)
        self.assertEqual(_parse_cp_text("cP 1500"), 1500)
        self.assertEqual(_parse_cp_text("CP-250"), 250)
        self.assertIsNone(_parse_cp_text("CP 7"))  # Single digit below 10 rejected
        self.assertIsNone(_parse_cp_text("No CP here"))
        self.assertIsNone(_parse_cp_text("CP 9999999"))


if __name__ == "__main__":
    unittest.main()
