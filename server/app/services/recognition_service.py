"""단어 구간 하나를 모델에 넣고 단어를 판정한다.

word_segment_service 가 만든 (60, 420) 을 받아 50개 클래스 중 하나를 고르고,
임계값을 넘지 못하면 아무 단어도 주장하지 않는다.

## 두 가지 기준

확신도 >= 0.5 이고 2위와의 격차 >= 0.15 여야 단어를 말한다.

기준값의 근거는 영상 5개(WORD0001, 5시점) 검증이다. 단어 단위 결과가
최저 확신도 0.615, 격차 최저 0.377 이었으므로 전 시점이 통과한다.

## 격차 기준이 실제로 작동하는 구간은 좁다

softmax 는 합이 1 이므로 2위는 아무리 커도 (1 - 확신도) 다. 따라서
격차는 최소 (2 x 확신도 - 1) 이고, 이것이 0.15 를 넘는 지점은
확신도 0.575 다. **즉 확신도가 0.575 이상이면 격차 기준은 자동으로
만족되어 아무 일도 하지 않는다.**

    확신도   2위 최대   최소 격차   격차 기준이 거절 가능?
    0.500    0.500     0.000      가능
    0.550    0.450     0.100      가능
    0.575    0.425     0.150      경계
    0.615    0.385     0.230      불가능
    0.900    0.100     0.800      불가능

실제로 걸러내는 구간은 확신도 [0.5, 0.575) 뿐이다. 검증 데이터의 최저값이
0.615 였으므로 **그 데이터는 이 기준을 한 번도 시험하지 않았다.** 0.15 라는
값 자체에는 근거가 없다.

그래도 남겨두는 이유는 둘이다. 확신도 기준을 나중에 낮추면 그때부터
의미를 갖고, 지금도 0.5~0.575 구간에서는 실제로 동작한다.

## 두 기준이 함께 놓치는 것

학습에 없는 동작(out-of-vocabulary)은 막지 못한다. 50-way softmax 는
처음 보는 입력에도 한 클래스에 0.9 를 몰아주는 일이 흔하다. 그러면 확신도도
격차도 통과하고 엉뚱한 단어가 안경에 뜬다. 이건 정렬된 확률만 봐서는
풀 수 없고, 거부 클래스나 특징 공간 거리 같은 다른 신호가 필요하다.
지금은 사용자가 구간을 직접 표시한다는 것에 기대고 있다.
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
        """디버깅용. 거절당한 경우 무엇이 1위였는지 남긴다.

        임계값을 조정할 때 보는 값이라 경계 근처가 중요하다. 4자리로 줄이면
        margin 0.14995 가 0.15 로 찍혀서 "기준과 같은데 거절됨"처럼 보인다.
        """
        return {
            "candidate": self.candidate,
            "class_index": self.class_index,
            "confidence": round(self.confidence, 6),
            "margin": round(self.margin, 6),
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

    if probabilities.shape[0] < 2:
        raise ValueError(
            f"Model must output at least 2 classes; got {probabilities.shape[0]}."
        )
    if not np.all(np.isfinite(probabilities)):
        # 가중치가 깨졌거나 발산한 모델. 그냥 두면 confidence 가 NaN 인 채로
        # JSON 에 실리고, 그 메시지는 표준 JSON 이 아니라 앱의 파싱이 통째로
        # 실패한다.
        raise ValueError("Model output contains NaN or Inf.")
    total = float(probabilities.sum())
    if not 0.99 <= total <= 1.01:
        # 마지막 층이 softmax 가 아니면(로짓, 로그확률) 임계값이 전부
        # 무의미해진다. 로짓이면 모든 구간이 통과하고, 로그확률이면 모든
        # 구간이 거절된다. 둘 다 조용히 일어난다.
        raise ValueError(
            f"Model output is not a probability distribution (sum={total:.4f})."
        )

    order = np.argsort(-probabilities, kind="stable")
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
