"""서버가 실제로 쓰는 경로 그대로 영상을 단어까지 돌린다.

verify_from_video.py 는 파이프라인을 재구성해서 확인하는 도구다. 이쪽은
서버 코드를 그대로 부른다 - word_store 에 프레임을 넣고, end_word 로 구간을
닫고, recognize_word_segment 로 판정한다. 미니앱이 버튼을 눌렀을 때
서버 안에서 벌어지는 일과 같다.

프레임을 버리는 방식이 중요하다. MediaPipe 가 30fps 를 못 따라가서 버리는데,
그 간격이 균일하지 않다. 균일하게만 흉내내면 시간축 리샘플이 할 일이 없어서
가장 좋은 경우만 재게 된다.

    uniform : N개마다 하나. 간격이 정확히 일정하다
    random  : 확률로 버림. 실제 latest-wins 큐에 가깝다
    stall   : 중간에 한 번 길게 멈춤. MediaPipe 가 밀렸을 때

    python scripts/verify_word_pipeline.py ../data/videos/*.mp4
    python scripts/verify_word_pipeline.py ../data/videos/*.mp4 --every 5
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

from app.config import (  # noqa: E402
    RECOGNITION_CONFIDENCE_THRESHOLD,
    RECOGNITION_MARGIN_THRESHOLD,
)
from app.services.label_service import LabelError, word_for_index  # noqa: E402
from app.services.mediapipe_service import get_mediapipe_service  # noqa: E402
from app.services.openpose_converter import convert_to_openpose  # noqa: E402
from app.services.recognition_model import (  # noqa: E402
    preload_recognition_model,
    recognition_model_error,
)
from app.services.recognition_service import (  # noqa: E402
    recognize_word_segment,
)
from app.services.word_segment_service import (  # noqa: E402
    build_openpose_feature_vector,
    word_store,
)

SOURCE_FPS = 30.0


def keep_uniform(total: int, every: int, rng) -> list[int]:
    return list(range(0, total, every))


def keep_random(total: int, every: int, rng) -> list[int]:
    kept = [i for i in range(total) if rng.random() < 1.0 / every]
    if kept and kept[0] != 0:
        kept.insert(0, 0)
    if kept and kept[-1] != total - 1:
        kept.append(total - 1)
    return kept


def keep_stall(total: int, every: int, rng) -> list[int]:
    """중간에서 0.5초 동안 아무것도 처리하지 못한 상황."""
    start = total // 2
    blocked = set(range(start, min(start + int(0.5 * SOURCE_FPS), total)))
    return [i for i in range(0, total, every) if i not in blocked]


DROP_MODES = {
    "uniform": keep_uniform,
    "random": keep_random,
    "stall": keep_stall,
}


def extract_all(path: Path) -> np.ndarray:
    """영상 전체를 411 좌표열로. MediaPipe 는 영상당 한 번만 돌린다."""
    service = get_mediapipe_service()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없다: {path}")

    declared_fps = capture.get(cv2.CAP_PROP_FPS)
    if not (SOURCE_FPS * 0.95 <= declared_fps <= SOURCE_FPS * 1.05):
        capture.release()
        raise SystemExit(
            f"{path.name}: 원본이 {declared_fps:.2f}fps 다. 이 스크립트는 "
            f"{SOURCE_FPS:.0f}fps 를 전제로 타임스탬프를 만든다."
        )

    rows = []
    started = time.time()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            rows.append(
                build_openpose_feature_vector(
                    convert_to_openpose(
                        service.extract_keypoints_from_image(image)
                    )
                )
            )
    finally:
        capture.release()

    elapsed = time.time() - started
    print(
        f"  MediaPipe {len(rows)}프레임 / {elapsed:.1f}초 "
        f"= {len(rows) / max(elapsed, 1e-6):.1f} fps"
    )
    return rows


def run_segment(session_id: str, frames, kept: list[int]):
    """word_start -> append -> end_word -> 판정. 서버 코드 그대로."""
    generation = word_store.start_session(session_id)
    word_store.start_word(session_id, generation)
    for index in kept:
        word_store.append(
            session_id,
            frames[index],
            index * 1000.0 / SOURCE_FPS,
            generation,
        )
    segment = word_store.end_word(session_id, generation)
    recognition = recognize_word_segment(segment.sequence)
    word_store.clear_session(session_id, generation)
    return segment, recognition


def expected_word(path: Path) -> str | None:
    name = path.stem
    if not name.startswith("WORD"):
        return None
    try:
        return word_for_index(int(name[4:8]) - 1)
    except (ValueError, LabelError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--every", type=int, default=3,
                        help="평균적으로 N개마다 하나만 처리 (기본 3 = 10fps)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.every < 1:
        parser.error(f"--every 는 1 이상이어야 한다: {args.every}")

    print("인식 모델 읽는 중...")
    if not preload_recognition_model():
        raise SystemExit(f"모델을 읽지 못했다: {recognition_model_error()}")
    print(
        f"임계값: 확신도 {RECOGNITION_CONFIDENCE_THRESHOLD} / "
        f"격차 {RECOGNITION_MARGIN_THRESHOLD}\n"
    )

    rows = []
    for video in args.videos:
        print(f"=== {video.name} ===")
        frames = extract_all(video)
        answer = expected_word(video)
        rng = np.random.default_rng(args.seed)

        for mode, chooser in DROP_MODES.items():
            kept = chooser(len(frames), args.every, rng)
            if len(kept) < 8:
                continue
            gaps = np.diff([i * 1000.0 / SOURCE_FPS for i in kept])
            segment, recognition = run_segment(
                f"verify-{video.stem}-{mode}", frames, kept
            )
            metadata = segment.metadata()
            rows.append({
                "mode": mode,
                "recognition": recognition,
                "correct": None if answer is None
                else recognition.candidate == answer,
            })
            verdict = "" if answer is None else (
                " 정답" if rows[-1]["correct"] else " 오답"
            )
            print(
                f"  [{mode:<7}] {len(kept):3d}프레임 "
                f"간격 {gaps.min():.0f}~{gaps.max():.0f}ms -> "
                f"복원 {metadata['uniform_frame_count']:3d}  "
                f"{recognition.candidate:<6} "
                f"{recognition.confidence:.3f} 격차 {recognition.margin:.3f}"
                f"{verdict}  {'말함' if recognition.text else '말 안 함'}"
            )
        print()

    print("=" * 70)
    print(f"요약 (every={args.every}, 평균 {SOURCE_FPS / args.every:.1f}fps)")
    print("=" * 70)
    for mode in DROP_MODES:
        picked = [r for r in rows if r["mode"] == mode]
        if not picked:
            continue
        correct = [r["correct"] for r in picked if r["correct"] is not None]
        spoke = [r for r in picked if r["recognition"].text]
        confidences = [r["recognition"].confidence for r in picked]
        print(
            f"  [{mode:<7}] 1위가 정답 {sum(correct)}/{len(correct)}  "
            f"임계값 통과 {len(spoke)}/{len(picked)}  "
            f"확신도 평균 {np.mean(confidences):.3f} 최저 {min(confidences):.3f}"
        )
    print("\n  '임계값 통과'가 사용자에게 실제로 단어가 보이는 횟수다.")
    print("  1위가 정답인데 통과 못 하면 인식 실패로 처리된다.")


if __name__ == "__main__":
    main()
