import os
import unittest
from src.classifier import classify_pokemon_from_image, is_stat_combination_possible

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
IMG_MIGHTYENA = os.path.join(SAMPLES_DIR, "cp25_mightyena.png")
IMG_FROAKIE = os.path.join(SAMPLES_DIR, "cp11_froakie.png")
IMG_PIKACHU = os.path.join(SAMPLES_DIR, "cp11_flying_pikachu_retina.png")
IMG_FLETCHLING = os.path.join(SAMPLES_DIR, "cp10_fletchling.png")


class TestClassifier(unittest.TestCase):
    def test_mightyena_disambiguation(self):
        """Verify that CP 25 + HP 17 + Poochyena Candy triangulates to Mightyena."""
        res = classify_pokemon_from_image(IMG_MIGHTYENA)
        self.assertEqual(res["species"], "Mightyena")
        self.assertEqual(res["cp"], 25)
        self.assertEqual(res["hp"], 17)
        self.assertEqual(res["candy_family"], "POOCHYENA")

    def test_math_combination_poochyena_vs_mightyena(self):
        """Verify mathematical impossibility of Poochyena having CP 25 and HP 17."""
        poochyena_stats = {"atk": 96, "def": 61, "sta": 111}
        mightyena_stats = {"atk": 171, "def": 132, "sta": 172}

        self.assertFalse(is_stat_combination_possible(poochyena_stats, target_cp=25, target_hp=17))
        self.assertTrue(is_stat_combination_possible(mightyena_stats, target_cp=25, target_hp=17))

    def test_froakie_classification(self):
        """Verify Froakie screenshot classification."""
        res = classify_pokemon_from_image(IMG_FROAKIE)
        self.assertEqual(res["species"], "Froakie")
        self.assertEqual(res["cp"], 11)
        self.assertEqual(res["hp"], 11)
        self.assertEqual(res["candy_family"], "FROAKIE")

    def test_pikachu_classification(self):
        """Verify Flying Pikachu screenshot classification."""
        res = classify_pokemon_from_image(IMG_PIKACHU)
        self.assertEqual(res["species"], "Pikachu")
        self.assertEqual(res["cp"], 11)
        self.assertEqual(res["hp"], 10)
        self.assertEqual(res["candy_family"], "PIKACHU")

    def test_fletchling_classification(self):
        """Verify Fletchling screenshot classification."""
        res = classify_pokemon_from_image(IMG_FLETCHLING)
        self.assertEqual(res["species"], "Fletchling")
        self.assertEqual(res["cp"], 10)
        self.assertEqual(res["hp"], 12)
        self.assertEqual(res["candy_family"], "FLETCHLING")

    def test_jigglypuff_disambiguation(self):
        """Verify that CP 27 + HP 42 + Jigglypuff Candy disambiguates to Jigglypuff over Igglybuff."""
        img_jiggly = os.path.join(SAMPLES_DIR, "cp27_jigglypuff.png")
        res = classify_pokemon_from_image(img_jiggly)
        self.assertEqual(res["species"], "Jigglypuff")
        self.assertEqual(res["cp"], 27)
        self.assertEqual(res["hp"], 42)
        self.assertEqual(res["candy_family"], "JIGGLYPUFF")
        self.assertIn("Igglybuff", res["candidates"])
        self.assertIn("Jigglypuff", res["candidates"])


if __name__ == "__main__":
    unittest.main()
