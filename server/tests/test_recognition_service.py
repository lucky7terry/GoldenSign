"""단어 판정 임계값.

확신도만 보면 안 된다. softmax 는 합이 1 이라 모르는 동작을 넣어도 어딘가에
확률이 몰리고, 두 클래스가 팽팽한 경우(0.45 대 0.43)와 하나가 확실한
경우를 구분하지 못한다. 2위와의 격차를 같이 본다.

기준값의 근거는 영상 5개(WORD0001, 5시점) 검증이다. 이 파일은 그 결론을
코드에 고정한다 - 단어 단위 결과는 통과하고, 슬라이딩 윈도우의 애매한
판정은 거절되어야 한다.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import (  # noqa: E402
    RECOGNITION_CONFIDENCE_THRESHOLD,
    RECOGNITION_MARGIN_THRESHOLD,
)
from app.services import recognition_service  # noqa: E402
from app.services.recognition_model import (  # noqa: E402
    FEATURE_DIM,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
)


def _probabilities(top: float, second: float, top_index: int = 0) -> np.ndarray:
    """1위와 2위를 지정하고 나머지에 남은 확률을 고르게 나눈다."""
    rest = (1.0 - top - second) / (NUM_CLASSES - 2)
    values = np.full(NUM_CLASSES, rest, dtype=np.float32)
    values[top_index] = top
    values[(top_index + 1) % NUM_CLASSES] = second
    return values


def _recognize(top: float, second: float, top_index: int = 0):
    sequence = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    predictor = mock.Mock(
        return_value=_probabilities(top, second, top_index)[None]
    )
    with mock.patch.object(
        recognition_service, "get_recognition_predictor", return_value=predictor
    ):
        return recognition_service.recognize_word_segment(sequence)


class ThresholdTest(unittest.TestCase):
    def test_validated_word_mode_result_is_accepted(self):
        """검증에서 가장 낮았던 값 - 확신도 0.615, 격차 0.377."""
        result = _recognize(0.615, 0.615 - 0.377)

        self.assertTrue(result.accepted)
        self.assertIsNotNone(result.text)
        self.assertEqual(result.text, result.candidate)

    def test_ambiguous_sliding_window_result_is_rejected(self):
        """슬라이딩 윈도우의 애매한 판정 - 확신도 0.366, 격차 0.083."""
        result = _recognize(0.366, 0.366 - 0.083)

        self.assertFalse(result.accepted)
        self.assertIsNone(result.text)
        # 무엇이 1위였는지는 남는다. 임계값을 조정할 때 필요하다.
        self.assertTrue(result.candidate)

    def test_high_confidence_but_close_second_is_rejected(self):
        """두 단어가 팽팽하면 확신도가 높아도 주장하지 않는다."""
        result = _recognize(0.48, 0.45)

        self.assertGreater(result.margin, 0.0)
        self.assertLess(result.margin, RECOGNITION_MARGIN_THRESHOLD)
        self.assertFalse(result.accepted)

    def test_clear_second_but_low_confidence_is_rejected(self):
        result = _recognize(0.30, 0.02)

        self.assertGreater(result.margin, RECOGNITION_MARGIN_THRESHOLD)
        self.assertLess(result.confidence, RECOGNITION_CONFIDENCE_THRESHOLD)
        self.assertFalse(result.accepted)

    def test_boundary_values_pass(self):
        """임계값과 정확히 같으면 통과한다(>= 이므로)."""
        result = _recognize(
            RECOGNITION_CONFIDENCE_THRESHOLD,
            RECOGNITION_CONFIDENCE_THRESHOLD - RECOGNITION_MARGIN_THRESHOLD,
        )

        self.assertTrue(result.accepted)

    def test_the_winning_class_is_reported_wherever_it_sits(self):
        result = _recognize(0.9, 0.05, top_index=37)

        self.assertEqual(result.class_index, 37)


class PayloadTest(unittest.TestCase):
    def test_public_payload_is_what_the_miniapp_reads(self):
        payload = _recognize(0.8, 0.05).public()

        self.assertEqual(set(payload), {"text", "confidence", "is_final"})
        self.assertTrue(payload["is_final"])

    def test_rejected_payload_has_no_text(self):
        payload = _recognize(0.2, 0.1).public()

        self.assertIsNone(payload["text"])
        # 확신도는 그대로 실어 보낸다. 앱이 "인식 실패"를 구분할 수 있다.
        self.assertGreater(payload["confidence"], 0.0)

    def test_detail_explains_a_rejection(self):
        detail = _recognize(0.2, 0.1).detail()

        self.assertFalse(detail["accepted"])
        self.assertIn("candidate", detail)
        self.assertEqual(
            detail["confidence_threshold"], RECOGNITION_CONFIDENCE_THRESHOLD
        )
        self.assertEqual(
            detail["margin_threshold"], RECOGNITION_MARGIN_THRESHOLD
        )


class ShapeTest(unittest.TestCase):
    def test_wrong_shape_is_rejected(self):
        predictor = mock.Mock()
        with mock.patch.object(
            recognition_service,
            "get_recognition_predictor",
            return_value=predictor,
        ):
            for shape in ((59, FEATURE_DIM), (SEQUENCE_LENGTH, 411), (60,)):
                with self.subTest(shape=shape):
                    with self.assertRaises(ValueError):
                        recognition_service.recognize_word_segment(
                            np.zeros(shape, dtype=np.float32)
                        )
        predictor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
