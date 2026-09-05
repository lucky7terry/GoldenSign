"""서버가 실제로 쓰는 경로 그대로 영상 하나를 단어까지 돌린다.

verify_from_video.py 는 파이프라인을 재구성해서 확인하는 도구다. 이쪽은
서버 코드를 그대로 부른다 - word_store 에 프레임을 넣고, end_word 로 구간을
닫고, recognize_word_segment 로 판정한다. 미니앱이 버튼을 눌렀을 때
서버 안에서 벌어지는 일과 같다.

    python scripts/verify_word_pipeline.py ../data/videos/*.mp4
    python scripts/verify_word_pipeline.py ../data/videos/WORD0001_REAL01_F.mp4 --every 1

--every 는 MediaPipe 가 못 따라가서 프레임이 버려지는 상황을 흉내낸다.
실제 서버는 12.7fps 쯤이므로 기본값 3(10fps)이 그에 가깝다.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import (  # noqa: E402
    RECOGNITION_CONFIDENCE_THRESHOLD,
    RECOGNITION_MARGIN_THRESHOLD,
)
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


def expected_word(path: Path) -> str | None:
    from app.services.label_service import LabelError, word_for_index

    name = path.stem
    if not name.startswith("WORD"):
        return None
    try:
        return word_for_index(int(name[4:8]) - 1)
    except (ValueError, LabelError):
        return None


def run_one(path: Path, every: int) -> dict:
    """word_start -> 프레임 -> word_end 를 서버 코드로 그대로 밟는다."""
    service = get_mediapipe_service()
    session_id = f"verify-{path.stem}"
    generation = word_store.start_session(session_id)
    word_store.start_word(session_id, generation)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없다: {path}")

    index = 0
    started = time.time()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index % every == 0:
                keypoints = convert_to_openpose(
                    service.extract_keypoints_from_image(image)
                )
                word_store.append(
                    session_id,
                    build_openpose_feature_vector(keypoints),
                    index * 1000.0 / SOURCE_FPS,
                    generation,
                )
            index += 1
    finally:
        capture.release()

    elapsed = time.time() - started
    segment = word_store.end_word(session_id, generation)
    recognition = recognize_word_segment(segment.sequence)
    word_store.clear_session(session_id, generation)

    return {
        "path": path,
        "metadata": segment.metadata(),
        "recognition": recognition,
        "seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--every", type=int, default=3,
                        help="N프레임마다 하나만 처리 (기본 3 = 10fps)")
    args = parser.parse_args()

    if args.every < 1:
        parser.error(f"--every 는 1 이상이어야 한다: {args.every}")

    print("인식 모델 읽는 중...")
    if not preload_recognition_model():
        raise SystemExit(
            f"모델을 읽지 못했다: {recognition_model_error()}"
        )
    print(
        f"임계값: 확신도 {RECOGNITION_CONFIDENCE_THRESHOLD} / "
        f"격차 {RECOGNITION_MARGIN_THRESHOLD}\n"
    )

    results = []
    for video in args.videos:
        outcome = run_one(video, args.every)
        results.append(outcome)

        metadata = outcome["metadata"]
        recognition = outcome["recognition"]
        answer = expected_word(video)
        verdict = "" if answer is None else (
            " 정답" if recognition.candidate == answer else " 오답"
        )
        served = "말함" if recognition.text else "말 안 함"

        print(f"{video.name}")
        print(
            f"  프레임 {metadata['frame_count']:3d} -> "
            f"30fps 복원 {metadata['uniform_frame_count']:3d} -> 60  "
            f"({metadata['span_ms']:.0f}ms, "
            f"시간축={metadata['resampled_on_time']})"
        )
        print(
            f"  {recognition.candidate:<8} 확신도 {recognition.confidence:.3f} "
            f"격차 {recognition.margin:.3f}{verdict}  -> 사용자에게 {served}"
        )
        print(f"  MediaPipe {outcome['seconds']:.1f}초\n")

    answered = [r for r in results if r["recognition"].text]
    correct = [
        r for r in results
        if expected_word(r["path"]) == r["recognition"].candidate
    ]
    print("=" * 60)
    print(f"1위가 정답:      {len(correct)}/{len(results)}")
    print(f"임계값 통과:     {len(answered)}/{len(results)}")
    if answered:
        lowest = min(r["recognition"].confidence for r in answered)
        print(f"통과한 것 중 최저 확신도: {lowest:.3f}")
    rejected = [r for r in results if not r["recognition"].text]
    for r in rejected:
        rec = r["recognition"]
        print(
            f"거절: {r['path'].name} — {rec.candidate} "
            f"확신도 {rec.confidence:.3f} 격차 {rec.margin:.3f}"
        )


if __name__ == "__main__":
    main()
