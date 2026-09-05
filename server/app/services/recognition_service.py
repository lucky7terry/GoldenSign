"""단어 구간 하나를 모델에 넣고 단어를 판정한다.

word_segment_service 가 만든 (60, 420) 을 받아 50개 클래스 중 하나를 고르고,
임계값을 넘지 못하면 아무 단어도 주장하지 않는다.

## 왜 확신도만 보면 안 되나

softmax 는 항상 합이 1 이라, 모델이 전혀 모르는 동작을 넣어도 어딘가에
확률이 몰린다. 50개가 고르게 나뉜 경우(1위 0.05)와 두 개가 팽팽한
경우(1위 0.45, 2위 0.43)는 성격이 다른데 확신도만으로는 구분되지 않는다.
그래서 2위와의 격차를 같이 본다.

기준값의 근거는 영상 5개(WORD0001, 5시점) 검증이다. 단어 단위 결과가
최저 확신도 0.615, 격차 최저 0.377 이었으므로 전 시점이 통과한다.
"""

import logging
from dataclasses import dataclass

import numpy as np

from app.config import (
    RECOGNITION_CONFIDENCE_THRESHOLD,
    RECOGNITION_MARGIN_THRESHOLD,
)
from app.services.label_service import word_for_index
from app.services.recognition_model import (
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    get_recognition_predictor,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WordRecognition:
    """한 구간의 판정 결과.

    text 는 임계값을 넘었을 때만 채운다. 넘지 못하면 None 이다 -
    자리표시자 문자열을 넣으면 그게 그대로 안경 화면에 뜬다.
    """

    text: str | None
    confidence: float
    margin: float
    class_index: int
    candidate: str
    accepted: bool

    def public(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "is_final": True,
        }

    def detail(self) -> dict:
        """디버깅용. 거절당한 경우 무엇이 1위였는지 남긴다."""
        return {
            "candidate": self.candidate,
            "class_index": self.class_index,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "accepted": self.accepted,
            "confidence_threshold": RECOGNITION_CONFIDENCE_THRESHOLD,
            "margin_threshold": RECOGNITION_MARGIN_THRESHOLD,
        }


def recognize_word_segment(sequence: np.ndarray) -> WordRecognition:
    """(SEQUENCE_LENGTH, FEATURE_DIM) 한 구간을 판정한다."""
    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
        raise ValueError(
            f"Word segment must be ({SEQUENCE_LENGTH}, {FEATURE_DIM}); "
            f"got {tuple(sequence.shape)}."
        )

    predictor = get_recognition_predictor()
    probabilities = np.asarray(
        predictor(sequence[None].astype(np.float32))
    )[0]

    order = np.argsort(-probabilities)
    class_index = int(order[0])
    confidence = float(probabilities[class_index])
    margin = confidence - float(probabilities[int(order[1])])
    candidate = word_for_index(class_index)

    accepted = (
        confidence >= RECOGNITION_CONFIDENCE_THRESHOLD
        and margin >= RECOGNITION_MARGIN_THRESHOLD
    )

    return WordRecognition(
        text=candidate if accepted else None,
        confidence=confidence,
        margin=margin,
        class_index=class_index,
        candidate=candidate,
        accepted=accepted,
    )
