"""얼굴 랜드마커가 왜 아무것도 못 찾는지 좁힌다.

verify_from_video.py 에서 face 검출률이 0.00 으로 나올 때 쓴다.
한 프레임을 여러 조건으로 돌려 어느 조건에서 잡히는지 본다.

    python scripts/diagnose_face_detection.py ../data/videos/WORD0001_REAL01_F.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def make_landmarker(min_confidence: float):
    return mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
    )


def detect(landmarker, image_bgr) -> int:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    )
    if not result.face_landmarks:
        return 0
    return len(result.face_landmarks[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame", type=int, default=None, help="기본: 영상 중간")
    args = parser.parse_args()

    print(f"모델 파일: {MODEL}  ({MODEL.stat().st_size:,} bytes)")

    capture = cv2.VideoCapture(str(args.video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    index = args.frame if args.frame is not None else total // 2
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit("프레임을 읽지 못했다")

    height, width = frame.shape[:2]
    print(f"프레임 {index}/{total}  {width}x{height}  dtype={frame.dtype} "
          f"channels={frame.shape[2]}")
    print()

    cases = []

    # 1) 원본, 임계값을 낮춰가며
    for confidence in (0.5, 0.3, 0.1):
        cases.append((f"원본 {width}x{height}, 임계 {confidence}",
                      frame, confidence))

    # 2) 축소 — 검출기가 큰 이미지에서 작은 얼굴을 놓치는 경우가 있다
    for scale in (0.5, 0.33):
        small = cv2.resize(frame, (int(width * scale), int(height * scale)))
        cases.append((f"축소 {small.shape[1]}x{small.shape[0]}, 임계 0.5",
                      small, 0.5))

    # 3) 인물 주변만 크롭 (가운데 세로 전체, 가로 절반)
    x0, x1 = width // 4, width * 3 // 4
    crop = frame[:, x0:x1]
    cases.append((f"중앙 크롭 {crop.shape[1]}x{crop.shape[0]}, 임계 0.5",
                  crop, 0.5))

    for label, image, confidence in cases:
        landmarker = make_landmarker(confidence)
        try:
            count = detect(landmarker, image)
        finally:
            landmarker.close()
        mark = f"검출 {count}점" if count else "검출 실패"
        print(f"  {label:<42} {mark}")

    print()
    print("읽는 법:")
    print("  전부 실패        -> 모델 파일이나 이미지 형식 문제")
    print("  임계값 낮추면 성공 -> mediapipe_service 의 min_face_* 를 낮춘다")
    print("  축소/크롭에서 성공 -> 검출기 입력 해상도 문제. 전처리로 축소해 넣는다")


if __name__ == "__main__":
    main()
