"""서버 전처리가 학습 파이프라인과 같은 결과를 내는지 확인한다.

학습 데이터 영상 폴더(OpenPose JSON 들)를 하나 골라, 서버의 feature_service
로 특징을 만들고 모델에 넣어 예측을 본다. 노트북의 predict_video_folder 와
같은 단어가 나와야 한다.

    python scripts/verify_model_pipeline.py <영상폴더> [--stride 8] [--topk 5]

영상 폴더는 학습 데이터의 WORD####_REAL##_# 디렉터리다. 폴더 이름이
WORD 로 시작하면 정답으로 보고 맞았는지까지 알려준다.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.feature_service import (  # noqa: E402
    FEATURE_DIM,
    RAW_DIM,
    SEQUENCE_LENGTH,
    build_features,
)
from app.services.label_service import word_for_index  # noqa: E402
from app.services.recognition_model import (  # noqa: E402
    load_recognition_model,
    make_predictor,
)


def load_openpose_folder(folder: Path) -> np.ndarray:
    """OpenPose JSON 폴더 -> (T, 411). 학습 노트북의 load_json 과 동일한 순서."""
    files = sorted(p for p in folder.iterdir() if p.suffix == ".json")
    if not files:
        raise SystemExit(f"JSON 파일이 없다: {folder}")

    frames = []
    for path in files:
        person = json.loads(path.read_text(encoding="utf-8"))["people"]
        feature = (
            person["pose_keypoints_2d"]
            + person["hand_left_keypoints_2d"]
            + person["hand_right_keypoints_2d"]
            + person["face_keypoints_2d"]
        )
        if len(feature) != RAW_DIM:
            raise SystemExit(f"{path.name}: {len(feature)}차원 (기대 {RAW_DIM})")
        frames.append(feature)
    return np.array(frames, dtype=np.float32)


def sliding_windows(features: np.ndarray, stride: int) -> np.ndarray:
    """노트북 make_eval_windows 와 같은 규칙."""
    if len(features) < SEQUENCE_LENGTH:
        # 짧으면 시간축을 늘려 한 개로 만든다.
        source = np.linspace(0, len(features) - 1, SEQUENCE_LENGTH)
        stretched = np.stack([
            np.interp(source, np.arange(len(features)), features[:, dim])
            for dim in range(FEATURE_DIM)
        ], axis=1)
        return stretched[None].astype(np.float32)

    starts = range(0, len(features) - SEQUENCE_LENGTH + 1, stride)
    return np.stack([features[s:s + SEQUENCE_LENGTH] for s in starts])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    raw = load_openpose_folder(args.folder)
    print(f"프레임 {len(raw)}개 x {raw.shape[1]}차원")

    features = build_features(raw)
    print(f"전처리 -> {features.shape}")
    print(f"  mean={features.mean():.3f}  std={features.std():.3f} "
          f"min={features.min():.2f} max={features.max():.2f}")
    print("  (학습 데이터 기준: mean 0.086  std 0.364)")

    windows = sliding_windows(features, args.stride)
    print(f"윈도우 {len(windows)}개 (stride={args.stride})")

    model = load_recognition_model()
    infer = make_predictor(model)

    # 노트북과 동일하게 윈도우 확률을 평균한다.
    probabilities = np.mean(
        [np.array(infer(w[None].astype(np.float32)))[0] for w in windows], axis=0
    )

    order = np.argsort(-probabilities)[: args.topk]
    print("\n예측:")
    for rank, index in enumerate(order, 1):
        print(f"  {rank}. {word_for_index(int(index)):<8} {probabilities[index]:.3f}")

    name = args.folder.name
    if name.startswith("WORD"):
        expected = int(name[4:8]) - 1
        predicted = int(order[0])
        mark = "일치" if predicted == expected else "불일치"
        print(f"\n정답 {word_for_index(expected)} / 예측 {word_for_index(predicted)} "
              f"-> {mark}")
        if predicted != expected:
            rank = int(np.where(np.argsort(-probabilities) == expected)[0][0]) + 1
            print(f"  정답의 순위: {rank}위 (확률 {probabilities[expected]:.3f})")


if __name__ == "__main__":
    main()
