"""영상 파일 하나로 서버 파이프라인 전 구간을 확인한다.

    영상 -> MediaPipe -> OpenPose 411 -> 특징 420 -> 모델 -> 단어

학습 데이터는 실제 OpenPose 로 만들어졌고 서버는 MediaPipe 를 OpenPose
규격으로 변환해 쓴다. 이 변환이 학습 분포와 얼마나 같은지는 지금까지
검증된 적이 없다. 원본 영상으로 돌려 정답 단어가 나오면 전 구간이 맞는다.

    python scripts/verify_from_video.py data/videos/WORD0001_REAL01_F.mp4

파일명이 WORD 로 시작하면 정답과 대조한다.
"""

import argparse
import os
import sys

# TensorFlow / MediaPipe 가 stderr 로 쏟는 C++ 로그를 줄인다. 반드시
# 그 라이브러리들을 임포트하기 전에 설정해야 한다. 이 스크립트의 출력은
# 사람이 읽는 것이라, 진단 수치가 로그에 묻히면 쓸모가 없다.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# Windows 기본 콘솔 인코딩(cp949)은 일부 기호를 못 쓴다. 파일로 리다이렉트하면
# 터미널과 달리 cp949 가 적용돼 출력 도중 죽는다. UTF-8 로 고정한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.feature_service import SEQUENCE_LENGTH, build_features  # noqa: E402
from app.services.label_service import word_for_index  # noqa: E402
from app.services.mediapipe_service import get_mediapipe_service  # noqa: E402
from app.services.openpose_converter import convert_to_openpose  # noqa: E402
from app.services.recognition_model import (  # noqa: E402
    load_recognition_model,
    make_predictor,
)


def frames_from_video(path: Path, every: int = 1):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없다: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index % every == 0:
                yield image
            index += 1
    finally:
        capture.release()
    print(f"  원본 fps={fps:.1f}, 총 {index}프레임")


def extract_openpose_sequence(path: Path, every: int) -> np.ndarray:
    service = get_mediapipe_service()
    rows = []
    started = time.time()
    for image in frames_from_video(path, every):
        keypoints = service.extract_keypoints_from_image(image)
        person = convert_to_openpose(keypoints).people
        rows.append(
            person.pose_keypoints_2d
            + person.hand_left_keypoints_2d
            + person.hand_right_keypoints_2d
            + person.face_keypoints_2d
        )
    elapsed = time.time() - started
    print(f"  MediaPipe: {len(rows)}프레임 / {elapsed:.1f}초 "
          f"= {len(rows)/max(elapsed, 1e-6):.1f} fps")
    return np.array(rows, dtype=np.float32)


def resample_time(features: np.ndarray, target: int = SEQUENCE_LENGTH) -> np.ndarray:
    """시간축을 target 길이로 리샘플한다.

    학습이 tf.image.resize(method="bilinear") 로 하는 것과 같은 좌표 규칙을
    쓴다(half-pixel centers). 출력 i 번째가 보는 원본 위치는
    (i + 0.5) * T / target - 0.5 다. np.linspace(0, T-1, target) 로 하면
    양 끝을 맞추는 다른 규칙이 되어 값이 조금씩 어긋난다.
    """
    count = len(features)
    positions = (np.arange(target) + 0.5) * (count / target) - 0.5
    positions = np.clip(positions, 0.0, count - 1.0)
    source = np.arange(count)
    return np.stack(
        [np.interp(positions, source, features[:, dim])
         for dim in range(features.shape[1])],
        axis=1,
    ).astype(np.float32)


def word_window(features: np.ndarray) -> np.ndarray:
    """구간 전체를 60프레임으로 리샘플해 윈도우 하나로 만든다.

    사용자가 한 단어의 시작과 끝을 표시하는 방식일 때의 서버 동작이다.
    학습 증강(crop_resample)이 가변 길이 구간을 60으로 늘리는 것과 같다.
    """
    return resample_time(features)[None]


def sliding_windows(features: np.ndarray, stride: int) -> np.ndarray:
    if len(features) < SEQUENCE_LENGTH:
        return word_window(features)
    starts = range(0, len(features) - SEQUENCE_LENGTH + 1, stride)
    return np.stack([features[s:s + SEQUENCE_LENGTH] for s in starts])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--every", type=int, default=1,
                        help="N프레임마다 하나씩 처리 (기본 1 = 전부)")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--word-mode", action="store_true",
                        help="구간 전체를 60프레임으로 리샘플해 한 번만 추론한다")
    args = parser.parse_args()

    print(f"[1/4] 영상 읽기 + MediaPipe: {args.video.name}")
    raw = extract_openpose_sequence(args.video, args.every)

    detected = (raw[:, 2::3] > 0.0).mean(axis=0)
    print(f"  411차원 통계: min={raw.min():.1f} max={raw.max():.1f} "
          f"mean={raw.mean():.1f}")
    print("  (학습 데이터: min=0.0 max=1202.9 mean=473.4)")
    print(f"  파트별 검출률: pose {detected[:25].mean():.2f} "
          f"L-hand {detected[25:46].mean():.2f} "
          f"R-hand {detected[46:67].mean():.2f} "
          f"face {detected[67:].mean():.2f}")

    print("\n[2/4] 특징 변환")
    features = build_features(raw)
    print(f"  {features.shape}  mean={features.mean():.3f} std={features.std():.3f}")
    print("  (학습 데이터: mean=0.086 std=0.364)")

    print("\n[3/4] 윈도우 + 추론")
    if args.word_mode:
        windows = word_window(features)
        seconds = len(raw) * args.every / 30.0
        print(f"  단어 단위: {len(features)}프레임 -> 60프레임으로 리샘플 (윈도우 1개)")
        print(f"  이 구간이 덮는 실제 시간: 약 {seconds:.1f}초")
        print("  (학습은 영상의 50~100% 구간을 60으로 리샘플해서 배웠다)")
    else:
        windows = sliding_windows(features, args.stride)
        print(f"  윈도우 {len(windows)}개 (stride={args.stride})")

    model = load_recognition_model()
    infer = make_predictor(model)
    per_window = np.stack(
        [np.array(infer(w[None].astype(np.float32)))[0] for w in windows]
    )

    probabilities = per_window.mean(axis=0)
    order = np.argsort(-probabilities)[: args.topk]

    print("\n[4/4] 결과 - 영상 전체 평균")
    for rank, index in enumerate(order, 1):
        print(f"  {rank}. {word_for_index(int(index)):<8} {probabilities[index]:.3f}")

    if len(per_window) > 1:
        winners = [int(w.argmax()) for w in per_window]
        print("\n  윈도우별 1위 분포:")
        for index in sorted(set(winners), key=winners.count, reverse=True)[:5]:
            print(f"    {word_for_index(index):<8} {winners.count(index)}/{len(winners)}")

    name = args.video.stem
    if name.startswith("WORD"):
        expected = int(name[4:8]) - 1
        predicted = int(order[0])
        print(f"\n  정답 {word_for_index(expected)} / 예측 "
              f"{word_for_index(predicted)} -> "
              f"{'일치' if predicted == expected else '불일치'}")
        if predicted != expected:
            rank = int(np.where(np.argsort(-probabilities) == expected)[0][0]) + 1
            print(f"  정답 순위 {rank}위 (확률 {probabilities[expected]:.3f})")


if __name__ == "__main__":
    main()
