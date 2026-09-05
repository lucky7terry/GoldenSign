"""단어 구간을 60프레임으로 만드는 세 가지 방법을 같은 영상으로 비교한다.

서버는 30fps 로 들어오는 영상을 MediaPipe 가 12.7fps 로만 따라가므로
프레임을 버린다. 그 버려진 간격을 어떻게 다룰지가 이 스크립트의 질문이다.

    옛 방식      : 도착한 프레임 그대로 build_features -> half-pixel 60
                   (간격이 불규칙하면 궤적이 시간축으로 일그러진다)
    복원 30fps   : 시간축 30fps 등간격으로 되돌린 뒤 build_features
                   (간격도 고치고 속도 배율도 학습과 같게 만든다)
    복원 관측fps : 시간축을 "그 구간이 실제로 도착한 평균 간격"으로 되돌린다
                   (간격만 고치고 프레임 수와 속도 배율은 그대로 둔다)

버리는 방식도 세 가지로 흉내낸다. 균일하게만 버리면 세 방법이 거의 같아서
차이가 안 보인다 - 실제 서버는 불규칙하게 버린다.

    uniform : N개마다 하나 (지금까지 하던 방식)
    random  : 확률로 버림 - 간격이 들쭉날쭉
    stall   : 중간에 한 번 길게 멈춤 - MediaPipe 가 밀렸을 때

MediaPipe 는 영상당 한 번만 돌린다(제일 느린 단계다).

    python scripts/compare_fps_restore.py ../data/videos/*.mp4
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.feature_service import (  # noqa: E402
    SEQUENCE_LENGTH,
    build_features,
)
from app.services.label_service import word_for_index  # noqa: E402
from app.services.mediapipe_service import get_mediapipe_service  # noqa: E402
from app.services.openpose_converter import convert_to_openpose  # noqa: E402
from app.services.recognition_model import (  # noqa: E402
    load_recognition_model,
    make_predictor,
)
from app.services.word_segment_service import (  # noqa: E402
    resample_index_half_pixel,
    resample_to_uniform_fps,
)

SOURCE_FPS = 30.0
METHODS = ("옛 방식", "복원 30fps", "복원 관측fps")


def extract_openpose_sequence(path: Path) -> np.ndarray:
    service = get_mediapipe_service()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없다: {path}")

    rows = []
    started = time.time()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            keypoints = service.extract_keypoints_from_image(image)
            person = convert_to_openpose(keypoints).people
            rows.append(
                person.pose_keypoints_2d
                + person.hand_left_keypoints_2d
                + person.hand_right_keypoints_2d
                + person.face_keypoints_2d
            )
    finally:
        capture.release()

    elapsed = time.time() - started
    print(
        f"  MediaPipe {len(rows)}프레임 / {elapsed:.1f}초 "
        f"= {len(rows) / max(elapsed, 1e-6):.1f} fps"
    )
    return np.array(rows, dtype=np.float32)


# --- 프레임을 버리는 방식 -------------------------------------------------


def keep_uniform(total: int, every: int, rng) -> list[int]:
    return list(range(0, total, every))


def keep_random(total: int, every: int, rng) -> list[int]:
    """확률로 버린다. 간격이 1~여러 프레임으로 들쭉날쭉해진다."""
    probability = 1.0 / every
    kept = [i for i in range(total) if rng.random() < probability]
    # 양 끝은 구간 경계라 살려둔다.
    if kept and kept[0] != 0:
        kept.insert(0, 0)
    if kept and kept[-1] != total - 1:
        kept.append(total - 1)
    return kept


def keep_stall(total: int, every: int, rng) -> list[int]:
    """중간에서 0.5초 동안 아무것도 처리하지 못한 상황."""
    stall_frames = int(0.5 * SOURCE_FPS)
    start = total // 2
    blocked = set(range(start, min(start + stall_frames, total)))
    return [i for i in range(0, total, every) if i not in blocked]


DROP_MODES = {
    "uniform": keep_uniform,
    "random": keep_random,
    "stall": keep_stall,
}


# --- 60프레임으로 만드는 세 가지 방법 -------------------------------------


def window_legacy(raw: np.ndarray, times_ms: list[float]) -> np.ndarray:
    return resample_index_half_pixel(build_features(raw), SEQUENCE_LENGTH)


def _restored(raw: np.ndarray, times_ms: list[float], fps: float) -> np.ndarray:
    uniform, on_time = resample_to_uniform_fps(
        [row.tolist() for row in raw], times_ms, fps
    )
    if not on_time:
        raise SystemExit("시간축 리샘플이 폴백으로 떨어졌다 - 스크립트 버그다.")
    return resample_index_half_pixel(build_features(uniform), SEQUENCE_LENGTH)


def window_restore_30(raw: np.ndarray, times_ms: list[float]) -> np.ndarray:
    return _restored(raw, times_ms, SOURCE_FPS)


def window_restore_observed(raw: np.ndarray, times_ms: list[float]) -> np.ndarray:
    """도착한 평균 간격으로 되돌린다. 균일 입력이면 항등이 된다."""
    span_seconds = (times_ms[-1] - times_ms[0]) / 1000.0
    observed_fps = (len(times_ms) - 1) / max(span_seconds, 1e-6)
    return _restored(raw, times_ms, observed_fps)


BUILDERS = {
    "옛 방식": window_legacy,
    "복원 30fps": window_restore_30,
    "복원 관측fps": window_restore_observed,
}


# --- 실행 -----------------------------------------------------------------


def expected_index(path: Path) -> int | None:
    name = path.stem
    if not name.startswith("WORD"):
        return None
    try:
        return int(name[4:8]) - 1
    except ValueError:
        return None


def summarize(probabilities: np.ndarray) -> tuple[int, float, float]:
    order = np.argsort(-probabilities)
    top = int(order[0])
    return (
        top,
        float(probabilities[top]),
        float(probabilities[top] - probabilities[order[1]]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--every", type=int, default=3,
                        help="평균적으로 N개마다 하나만 처리 (기본 3 = 10fps)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("모델 읽는 중...")
    infer = make_predictor(load_recognition_model())
    rows = []

    for video in args.videos:
        print(f"\n=== {video.name} ===")
        full = extract_openpose_sequence(video)
        answer = expected_index(video)
        rng = np.random.default_rng(args.seed)

        # 기준선: 원본 30fps 전부
        base_times = [i * 1000.0 / SOURCE_FPS for i in range(len(full))]
        top, confidence, _ = summarize(
            np.array(infer(window_legacy(full, base_times)[None]))[0]
        )
        print(f"  [기준] 30fps 전부 {len(full)}프레임  "
              f"{word_for_index(top):<8} 확신도 {confidence:.3f}")

        for mode, chooser in DROP_MODES.items():
            kept = chooser(len(full), args.every, rng)
            if len(kept) < 8:
                continue
            raw = full[kept]
            times = [i * 1000.0 / SOURCE_FPS for i in kept]
            gaps = np.diff(times)
            print(f"\n  [{mode}] {len(kept)}프레임  "
                  f"간격 평균 {gaps.mean():.1f}ms "
                  f"최소 {gaps.min():.1f} 최대 {gaps.max():.1f}ms")

            for method, builder in BUILDERS.items():
                probabilities = np.array(infer(builder(raw, times)[None]))[0]
                top, confidence, margin = summarize(probabilities)
                correct = None if answer is None else (top == answer)
                rows.append({
                    "mode": mode, "method": method,
                    "confidence": confidence, "margin": margin,
                    "correct": correct,
                })
                mark = "" if correct is None else (" 정답" if correct else " 오답")
                print(f"    {method:<12} {word_for_index(top):<8} "
                      f"확신도 {confidence:.3f}  격차 {margin:.3f}{mark}")

    print("\n" + "=" * 66)
    print(f"요약  (every={args.every}, 평균 {SOURCE_FPS / args.every:.1f}fps)")
    print("=" * 66)
    for mode in DROP_MODES:
        print(f"\n  [{mode}]")
        for method in METHODS:
            picked = [r for r in rows if r["mode"] == mode and r["method"] == method]
            if not picked:
                continue
            confidences = [r["confidence"] for r in picked]
            correct = [r["correct"] for r in picked if r["correct"] is not None]
            accuracy = f"{sum(correct)}/{len(correct)}" if correct else "-"
            print(f"    {method:<12} 정답 {accuracy}  "
                  f"확신도 평균 {np.mean(confidences):.3f} "
                  f"최저 {min(confidences):.3f}")

    print("\n  uniform 에서는 세 방법이 거의 같아야 한다(간격이 이미 균일하다).")
    print("  random / stall 에서 갈리는지가 이 비교의 핵심이다.")


if __name__ == "__main__":
    main()
