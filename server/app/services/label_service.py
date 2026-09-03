"""모델 클래스 인덱스 -> 한국어 단어.

모델 출력은 50차원 softmax이고, 인덱스 i가 WORD%04d(i+1)에 대응한다.
(학습 노트북의 `"WORD%04d" % (i + 1)` 규칙과 동일)
"""

import json
from functools import lru_cache
from pathlib import Path

NUM_CLASSES = 50

_LABELS_PATH = Path(__file__).resolve().parents[1] / "resources" / "word_labels.json"


class LabelError(RuntimeError):
    """라벨 파일이 없거나 모델 클래스 수와 맞지 않을 때."""


@lru_cache(maxsize=1)
def _labels() -> tuple[str, ...]:
    if not _LABELS_PATH.exists():
        raise LabelError(f"Word label file not found: {_LABELS_PATH}")

    mapping = json.loads(_LABELS_PATH.read_text(encoding="utf-8"))
    words = []
    for index in range(NUM_CLASSES):
        key = "WORD%04d" % (index + 1)
        if key not in mapping:
            raise LabelError(f"Missing label for {key} in {_LABELS_PATH}")
        words.append(mapping[key])
    return tuple(words)


def word_for_index(index: int) -> str:
    words = _labels()
    if not 0 <= index < len(words):
        raise LabelError(f"Class index out of range: {index}")
    return words[index]


def word_code_for_index(index: int) -> str:
    if not 0 <= index < NUM_CLASSES:
        raise LabelError(f"Class index out of range: {index}")
    return "WORD%04d" % (index + 1)


def all_words() -> tuple[str, ...]:
    return _labels()
