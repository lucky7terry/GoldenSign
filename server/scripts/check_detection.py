"""아무 영상이나 서버 경로로 돌려 부위별 검출률을 본다.

얼굴 크롭 수정이 데이터셋과 안경 영상 양쪽에서 되는지 확인하는 용도다.
둘은 조건이 정반대라 한쪽만 보면 안 된다.

    데이터셋: 전신, 얼굴이 화면 폭의 5.7%  -> 크롭 없이는 검출 0
    안경    : 근접, 얼굴이 화면을 크게 차지 -> 크롭이 너무 좁지 않은지

사용법:

    python scripts/check_detection.py ../data/videos/WORD0001_REAL01_F.mp4
    python scripts/check_detection.py ../data/glasses.mp4 --every 3
    python scripts/check_detection.py <영상> --no-crop     # 수정 전과 비교
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import mediapipe_service as mp_service  # noqa: E402
from app.services.openpose_converter import convert_to_openpose  # noqa: E402

PARTS = (
    ("pose", 0, 25),
    ("왼손", 25, 46),
    ("오른손", 46, 67),
    ("얼굴", 67, 137),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--every", type=int, default=1,
                        help="N프레임마다 하나만. 긴 영상을 빨리 볼 때")
    parser.add_argument("--no-crop", action="store_true",
                        help="얼굴 크롭을 끄고 원본을 넣는다 (수정 전 동작)")
    args = parser.parse_args()

    if args.no_crop:
        # 크롭 상자를 항상 None 으로 만들면 호출자가 원본 경로로 떨어진다.
        mp_service.MediaPipeService._face_crop_box = staticmethod(
            lambda *a, **k: None
        )
        print("얼굴 크롭 끔 — 수정 전 동작")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없다: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.video.name}  {width}x{height}  {fps:.1f}fps  {total}프레임")

    service = mp_service.get_mediapipe_service()
    rows: list[list[float]] = []
    crop_sides: list[int] = []
    index = 0
    started = time.time()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index % args.every == 0:
                keypoints = service.extract_keypoints_from_image(image)
                person = convert_to_openpose(keypoints).people
                rows.append(
                    person.pose_keypoints_2d
                    + person.hand_left_keypoints_2d
                    + person.hand_right_keypoints_2d
                    + person.face_keypoints_2d
                )
            index += 1
    finally:
        capture.release()

    if not rows:
        raise SystemExit("프레임을 하나도 못 읽었다")

    frames = np.asarray(rows, dtype=np.float32)
    elapsed = time.time() - started
    conf = frames[:, 2::3]
    xy = frames.reshape(len(frames), 137, 3)[..., :2]

    print(f"\n{len(frames)}프레임 / {elapsed:.1f}초 "
          f"= {len(frames)/max(elapsed,1e-6):.1f} fps\n")
    print(f"{'부위':<8}{'검출률':>8}{'좌표 0 아닌 비율':>18}")
    for name, lo, hi in PARTS:
        # 손·얼굴은 visibility 가 없어 신뢰도가 항상 1.0 이다. 그래서
        # 검출률만 보면 안 되고 좌표가 실제로 채워졌는지를 같이 본다.
        print(f"{name:<8}{conf[:, lo:hi].mean():>8.2f}"
              f"{(xy[:, lo:hi] != 0).mean():>18.2f}")

    # 언제 잡혔는지. 검출률만 보면 "3분의 1"이 수어 구간에 몰린 것인지
    # 영상 전체에 흩어진 것인지 구분할 수 없다. 앞뒤 대기 시간이 붙은
    # 영상은 전자가 정상이고, 후자면 검출이 불안정하다는 뜻이다.
    print()
    print("시간축 검출 (한 칸 = 프레임 하나, 긴 영상은 압축)")
    width_cells = min(len(frames), 72)
    step = len(frames) / width_cells
    for name, lo, hi in PARTS:
        present = (xy[:, lo:hi] != 0).any(axis=(1, 2))
        cells = []
        for cell in range(width_cells):
            start = int(cell * step)
            end = max(start + 1, int((cell + 1) * step))
            span = present[start:end]
            cells.append("#" if span.all() else ("+" if span.any() else "."))
        print(f"  {name:<6}{''.join(cells)}")
    seconds = len(frames) / (fps / max(args.every, 1)) if fps > 0 else 0.0
    if seconds:
        print(f"  {'':<6}0초{' ' * (width_cells - 8)}{seconds:.1f}초")

    face_xy = xy[:, 67:137]
    detected = (face_xy != 0).any(axis=(1, 2))
    print(f"\n얼굴이 잡힌 프레임: {int(detected.sum())}/{len(frames)} "
          f"({detected.mean():.1%})")
    if detected.any():
        found = face_xy[detected]
        print(f"  얼굴 좌표 범위 x[{found[..., 0].min():.0f}, "
              f"{found[..., 0].max():.0f}] "
              f"y[{found[..., 1].min():.0f}, {found[..., 1].max():.0f}]")
        print(f"  화면 크기      x[0, {width}] y[0, {height}]")
        # 복원을 빼먹으면 좌표가 좌상단에 몰린다. 그 사고를 여기서 잡는다.
        if found[..., 0].max() < width * 0.25 and found[..., 1].max() < height * 0.25:
            print("  ⚠ 얼굴 좌표가 좌상단에 몰려 있다 — 크롭 좌표 복원을 의심할 것")


if __name__ == "__main__":
    main()
