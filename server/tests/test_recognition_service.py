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
    assert top > rest, "top 이 나머지보다 작으면 argmax 가 딴 데로 간다"
    # float32 로 만들면 경계값 테스트가 반올림 때문에 우연히 통과한다.
    values = np.full(NUM_CLASSES, rest, dtype=np.float64)
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

    def test_margin_alone_can_reject(self):
        """격차 기준이 단독으로 일하는 유일한 구간을 고정한다.

        softmax 는 합이 1 이라 2위는 아무리 커도 (1 - 확신도) 다. 그래서
        확신도가 0.575 를 넘으면 격차는 자동으로 0.15 를 넘는다. 격차 기준이
        실제로 거절할 수 있는 구간은 확신도 [0.5, 0.575) 뿐이다.

        이 테스트가 없으면 accepted 에서 격차 검사를 통째로 지워도 전체
        테스트가 통과한다.
        """
        rejected = _recognize(0.52, 0.40)      # 확신도는 통과, 격차 0.12
        accepted = _recognize(0.52, 0.36)      # 확신도 같음, 격차 0.16

        self.assertGreaterEqual(
            rejected.confidence, RECOGNITION_CONFIDENCE_THRESHOLD
        )
        self.assertLess(rejected.margin, RECOGNITION_MARGIN_THRESHOLD)
        self.assertFalse(rejected.accepted)

        self.assertTrue(accepted.accepted)

    def test_margin_rule_is_inert_above_the_crossover(self):
        """확신도 0.575 위에서는 격차 기준이 아무 일도 하지 않는다.

        2위를 최대로(=1-확신도) 밀어도 격차 기준을 넘는다. 문서에 적은
        내용이 실제로 그런지 확인한다.
        """
        confidence = 0.62
        result = _recognize(confidence, 1.0 - confidence - 1e-9)

        self.assertGreaterEqual(result.margin, RECOGNITION_MARGIN_THRESHOLD)
        self.assertTrue(result.accepted)

    def test_just_above_and_just_below_the_confidence_threshold(self):
        """정확히 같은 값에 기대면 부동소수 반올림으로 우연히 통과한다.

        임계값을 환경변수로 바꾸면 그 우연이 깨진다. 아슬아슬하게 위/아래를
        본다.
        """
        epsilon = 1e-6
        clear_second = RECOGNITION_CONFIDENCE_THRESHOLD - 0.4

        above = _recognize(RECOGNITION_CONFIDENCE_THRESHOLD + epsilon, clear_second)
        below = _recognize(RECOGNITION_CONFIDENCE_THRESHOLD - epsilon, clear_second)

        self.assertTrue(above.accepted)
        self.assertFalse(below.accepted)

    def test_the_winning_class_is_reported_wherever_it_sits(self):
        result = _recognize(0.9, 0.05, top_index=37)

        self.assertEqual(result.class_index, 37)

    def test_the_label_comes_from_the_winning_class(self):
        """1위 인덱스로 라벨을 찾는지. 2위로 바꿔도 통과하면 안 된다."""
        sequence = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        predictor = mock.Mock(return_value=_probabilities(0.9, 0.05, 37)[None])

        with mock.patch.object(
            recognition_service, "get_recognition_predictor",
            return_value=predictor,
        ), mock.patch.object(
            recognition_service, "word_for_index", return_value="테스트단어"
        ) as word_for_index:
            result = recognition_service.recognize_word_segment(sequence)

        word_for_index.assert_called_once_with(37)
        self.assertEqual(result.text, "테스트단어")


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


class MalformedOutputTest(unittest.TestCase):
    """마지막 층이 softmax 가 아니면 임계값이 전부 무의미해진다.

    로짓이면 모든 구간이 통과하고, 로그확률이면 모든 구간이 거절된다.
    둘 다 조용히 일어나서 "모델이 이상하다"로 오진하기 쉽다.
    """

    @staticmethod
    def _run(values: np.ndarray):
        sequence = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        predictor = mock.Mock(return_value=values[None])
        with mock.patch.object(
            recognition_service, "get_recognition_predictor",
            return_value=predictor,
        ):
            return recognition_service.recognize_word_segment(sequence)

    def test_logits_are_rejected(self):
        values = np.full(NUM_CLASSES, -3.0)
        values[0], values[1] = 12.0, 8.0

        with self.assertRaises(ValueError):
            self._run(values)

    def test_log_probabilities_are_rejected(self):
        values = np.log(_probabilities(0.9, 0.05))

        with self.assertRaises(ValueError):
            self._run(values)

    def test_nan_output_is_rejected(self):
        values = _probabilities(0.9, 0.05)
        values[5] = np.nan

        with self.assertRaises(ValueError):
            self._run(values)

    def test_single_class_output_is_rejected(self):
        with self.assertRaises(ValueError):
            self._run(np.array([1.0]))

    def test_a_tie_resolves_deterministically(self):
        values = np.zeros(NUM_CLASSES)
        values[3] = values[41] = 0.5

        first = self._run(values)
        second = self._run(values)

        self.assertEqual(first.class_index, second.class_index)
        self.assertFalse(first.accepted)


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
