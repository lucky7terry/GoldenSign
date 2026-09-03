import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.label_service import (
    NUM_CLASSES,
    LabelError,
    all_words,
    word_code_for_index,
    word_for_index,
)


class LabelServiceTest(unittest.TestCase):
    def test_every_model_class_has_a_word(self):
        self.assertEqual(len(all_words()), NUM_CLASSES)
        self.assertTrue(all(word for word in all_words()))

    def test_index_zero_maps_to_first_word(self):
        self.assertEqual(word_code_for_index(0), "WORD0001")
        self.assertEqual(word_for_index(0), "배")

    def test_last_index_maps_to_last_word(self):
        self.assertEqual(word_code_for_index(NUM_CLASSES - 1), "WORD0050")
        self.assertEqual(word_for_index(NUM_CLASSES - 1), "갑자기")

    def test_out_of_range_index_is_rejected(self):
        for index in (-1, NUM_CLASSES):
            with self.assertRaises(LabelError):
                word_for_index(index)


if __name__ == "__main__":
    unittest.main()
