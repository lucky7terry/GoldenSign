"""OpenPose 411차원 -> 모델 입력 420차원 변환.

학습 노트북(transformer_tuning_얼굴포함_개선.ipynb)의 build_features()를
서버용으로 옮긴 것이다. 모델은 원본 픽셀 좌표가 아니라 신체 기준으로
정규화된 특징을 받으므로, 이 변환이 학습 때와 어긋나면 정확도가 무너진다.

레이아웃 (420 = 208 + 208 + 4):
    [위치 208 = 104점 x (x,y)][속도 208][파트별 검출률 4]

    pose  : UPPER_POSE 15점   anchor=목,     scale=어깨너비
    lhand : 손목 제외 20점     anchor=손목,   scale=손 span
    rhand : 손목 제외 20점
    face  : FACE_REDUCED 49점 anchor=코끝,   scale=눈 간격

학습과 다른 점 — 스트리밍 인과성:
    학습은 영상 전체를 보고 스케일 중앙값과 결측 보간을 계산한다.
    실시간에는 미래 프레임이 없으므로 여기서는 "지금 윈도우" 안에서만
    계산한다. 추론도 60프레임 윈도우 단위이므로 윈도우가 곧 영상 역할을
    한다. 영상 전체 대비 스케일이 조금 달라질 수 있으니, 정확도 문제가
    생기면 이 부분을 먼저 의심할 것.
"""

import numpy as np

RAW_DIM = 411
FEATURE_DIM = 420
SEQUENCE_LENGTH = 60

# BODY_25 상반신 15점. 서버가 채우지 않는 하반신 관절과 겹치지 않는다.
UPPER_POSE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18)

# OpenPose face 70점 중 수어에 의미 있는 49점 (눈썹/눈/코/입/동공).
FACE_REDUCED = (
    tuple(range(17, 27))
    + tuple(range(36, 48))
    + (27, 30, 31, 33, 35)
    + tuple(range(48, 68))
    + (68, 69)
)

CONF_THRESHOLD = 0.05
CLIP_POSITION = 6.0
CLIP_VELOCITY = 2.0

# 기준점 / 스케일에 쓰는 인덱스
POSE_NECK = 1
POSE_RIGHT_SHOULDER = 2
POSE_LEFT_SHOULDER = 5
POSE_MID_HIP = 8
FACE_NOSE_TIP = 30
FACE_RIGHT_EYE_CORNER = 36
FACE_LEFT_EYE_CORNER = 45

_FALLBACK_BODY_SCALE = 100.0
_HAND_SCALE_RATIO = 0.35
_FACE_SCALE_RATIO = 0.35


class FeatureShapeError(ValueError):
    """입력 프레임의 모양이 411차원 규격과 다를 때."""


def _split_parts(frames: np.ndarray):
    count = frames.shape[0]
    return (
        frames[:, :75].reshape(count, 25, 3),
        frames[:, 75:138].reshape(count, 21, 3),
        frames[:, 138:201].reshape(count, 21, 3),
        frames[:, 201:411].reshape(count, 70, 3),
    )


def _interpolate_missing(keypoints: np.ndarray):
    """confidence가 낮은 프레임의 좌표를 시간축 선형보간으로 채운다.

    한 번도 검출되지 않은 keypoint는 NaN으로 두고 나중에 0으로 만든다.
    (0,0)을 그대로 쓰면 '화면 왼쪽 위'라는 가짜 신호가 되기 때문이다.
    """
    positions = keypoints[..., :2].astype(np.float64).copy()
    valid = keypoints[..., 2] > CONF_THRESHOLD
    count, points, _ = positions.shape
    times = np.arange(count)

    for point in range(points):
        seen = valid[:, point]
        if seen.sum() == 0:
            positions[:, point, :] = np.nan
        elif seen.sum() < count:
            for axis in range(2):
                positions[:, point, axis] = np.interp(
                    times, times[seen], positions[seen, point, axis]
                )
    return positions, valid


def _robust_scale(distances: np.ndarray, usable: np.ndarray, fallback: float) -> float:
    if usable.sum() > 0:
        scale = float(np.median(distances[usable]))
        if np.isfinite(scale) and scale > 1e-3:
            return scale
    return fallback


def _body_anchor_and_scale(pose_xy: np.ndarray, pose_valid: np.ndarray):
    neck = pose_xy[:, POSE_NECK, :]
    mid_shoulder = 0.5 * (
        pose_xy[:, POSE_RIGHT_SHOULDER, :] + pose_xy[:, POSE_LEFT_SHOULDER, :]
    )
    anchor = np.where(pose_valid[:, POSE_NECK : POSE_NECK + 1], neck, mid_shoulder)
    anchor = np.where(np.isfinite(anchor), anchor, np.nanmean(pose_xy, axis=1))

    shoulder_width = np.linalg.norm(
        pose_xy[:, POSE_RIGHT_SHOULDER, :] - pose_xy[:, POSE_LEFT_SHOULDER, :],
        axis=-1,
    )
    scale = _robust_scale(
        shoulder_width,
        pose_valid[:, POSE_RIGHT_SHOULDER] & pose_valid[:, POSE_LEFT_SHOULDER],
        np.nan,
    )
    if not np.isfinite(scale):
        # 어깨가 안 잡히면 목~골반 길이로 대체한다.
        torso = np.linalg.norm(
            pose_xy[:, POSE_NECK, :] - pose_xy[:, POSE_MID_HIP, :], axis=-1
        )
        scale = _robust_scale(
            torso,
            pose_valid[:, POSE_NECK] & pose_valid[:, POSE_MID_HIP],
            _FALLBACK_BODY_SCALE,
        )
    return anchor, scale


def _hand_block(hand_xy: np.ndarray, hand_valid: np.ndarray, body_scale: float):
    """손목을 원점으로 한 손 모양. 손목 자신은 항상 0이므로 제외한다."""
    span = np.linalg.norm(hand_xy - hand_xy[:, 0:1, :], axis=-1).max(axis=1)
    scale = _robust_scale(
        span, hand_valid.any(axis=1), body_scale * _HAND_SCALE_RATIO
    )
    return (hand_xy[:, 1:, :] - hand_xy[:, 0:1, :]) / scale


def _face_block(face_xy: np.ndarray, face_valid: np.ndarray, body_scale: float):
    eye_distance = np.linalg.norm(
        face_xy[:, FACE_RIGHT_EYE_CORNER, :] - face_xy[:, FACE_LEFT_EYE_CORNER, :],
        axis=-1,
    )
    scale = _robust_scale(
        eye_distance,
        face_valid[:, FACE_RIGHT_EYE_CORNER] & face_valid[:, FACE_LEFT_EYE_CORNER],
        body_scale * _FACE_SCALE_RATIO,
    )
    selected = face_xy[:, FACE_REDUCED, :]
    return (selected - face_xy[:, FACE_NOSE_TIP : FACE_NOSE_TIP + 1, :]) / scale


def build_features(frames: np.ndarray) -> np.ndarray:
    """(T, 411) 원본 OpenPose 좌표 -> (T, 420) 모델 입력 특징.

    frames의 각 행은 pose(75) + left_hand(63) + right_hand(63) + face(210)
    순서의 [x, y, confidence] 나열이며, 좌표는 픽셀 단위다.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != RAW_DIM:
        raise FeatureShapeError(
            f"Expected (T, {RAW_DIM}) frames, got {frames.shape}."
        )
    if frames.shape[0] == 0:
        raise FeatureShapeError("Expected at least one frame.")

    pose, left_hand, right_hand, face = _split_parts(frames)
    pose_xy, pose_valid = _interpolate_missing(pose)
    left_xy, left_valid = _interpolate_missing(left_hand)
    right_xy, right_valid = _interpolate_missing(right_hand)
    face_xy, face_valid = _interpolate_missing(face)

    anchor, body_scale = _body_anchor_and_scale(pose_xy, pose_valid)

    blocks = [
        (pose_xy[:, UPPER_POSE, :] - anchor[:, None, :]) / body_scale,
        _hand_block(left_xy, left_valid, body_scale),
        _hand_block(right_xy, right_valid, body_scale),
        _face_block(face_xy, face_valid, body_scale),
    ]

    positions = np.concatenate(
        [block.reshape(block.shape[0], -1) for block in blocks], axis=1
    )
    positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
    positions = np.clip(positions, -CLIP_POSITION, CLIP_POSITION)

    velocity = np.clip(
        np.diff(positions, axis=0, prepend=positions[:1]),
        -CLIP_VELOCITY,
        CLIP_VELOCITY,
    )

    # 파트별 검출률 — 미검출 구간을 모델이 알 수 있게 한다.
    detection_flags = np.stack(
        [
            pose_valid.mean(axis=1),
            left_valid.mean(axis=1),
            right_valid.mean(axis=1),
            face_valid.mean(axis=1),
        ],
        axis=1,
    )

    features = np.concatenate([positions, velocity, detection_flags], axis=1)
    return features.astype(np.float32)
