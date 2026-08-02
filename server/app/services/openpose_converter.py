from typing import Any

from app.schemas.keypoint import ExtractedKeypoints, Landmark
from app.schemas.openpose import OpenPosePerson, OpenPoseResult

POSE_MP_TO_BODY25_IDX = [
    0,
    -1,
    12,
    14,
    16,
    11,
    13,
    15,
    -2,
    24,
    -3,
    -3,
    23,
    -3,
    -3,
    5,
    2,
    8,
    7,
    -3,
    -3,
    -3,
    -3,
    -3,
    -3,
]

FACE_KEYPOINT_COUNT = 70
HAND_KEYPOINT_COUNT = 21


def _landmark_confidence(landmark: Landmark) -> float:
    if landmark.visibility is not None:
        return float(landmark.visibility)
    return 1.0


def _zero_keypoints_2d(count: int) -> list[float]:
    return [0.0] * count * 3


def _zero_keypoints_3d(count: int) -> list[float]:
    return [0.0] * count * 4


def _to_2d_values(
    landmark: Landmark,
    image_width: int | None,
    image_height: int | None,
) -> list[float]:
    x = landmark.x * image_width if image_width is not None else landmark.x
    y = landmark.y * image_height if image_height is not None else landmark.y
    return [
        float(x),
        float(y),
        _landmark_confidence(landmark),
    ]


def _to_3d_values(landmark: Landmark) -> list[float]:
    return [
        float(landmark.x),
        float(landmark.y),
        float(landmark.z),
        _landmark_confidence(landmark),
    ]


def _average_landmark(
    first: Landmark | None,
    second: Landmark | None,
) -> Landmark | None:
    if first is None or second is None:
        return None

    first_visibility = _landmark_confidence(first)
    second_visibility = _landmark_confidence(second)
    return Landmark(
        x=(first.x + second.x) / 2,
        y=(first.y + second.y) / 2,
        z=(first.z + second.z) / 2,
        visibility=(first_visibility + second_visibility) / 2,
    )


def _pose_landmark_for_body25_index(
    pose: list[Landmark],
    mapping_index: int,
) -> Landmark | None:
    if mapping_index == -1:
        return _average_landmark(
            pose[11] if len(pose) > 11 else None,
            pose[12] if len(pose) > 12 else None,
        )
    if mapping_index == -2:
        return _average_landmark(
            pose[23] if len(pose) > 23 else None,
            pose[24] if len(pose) > 24 else None,
        )
    if mapping_index < 0 or mapping_index >= len(pose):
        return None
    return pose[mapping_index]


def _pose_to_body25_2d(keypoints: ExtractedKeypoints) -> list[float]:
    values: list[float] = []
    for mapping_index in POSE_MP_TO_BODY25_IDX:
        landmark = _pose_landmark_for_body25_index(
            keypoints.pose,
            mapping_index,
        )
        if landmark is None:
            values.extend([0.0, 0.0, 0.0])
        else:
            values.extend(
                _to_2d_values(
                    landmark,
                    keypoints.image_width,
                    keypoints.image_height,
                )
            )
    return values


def _pose_to_body25_3d(keypoints: ExtractedKeypoints) -> list[float]:
    values: list[float] = []
    for mapping_index in POSE_MP_TO_BODY25_IDX:
        landmark = _pose_landmark_for_body25_index(
            keypoints.pose,
            mapping_index,
        )
        if landmark is None:
            values.extend([0.0, 0.0, 0.0, 0.0])
        else:
            values.extend(_to_3d_values(landmark))
    return values


def _hand_to_openpose_2d(
    hand: list[Landmark],
    image_width: int | None,
    image_height: int | None,
) -> list[float]:
    if not hand:
        return _zero_keypoints_2d(HAND_KEYPOINT_COUNT)

    values: list[float] = []
    for index in range(HAND_KEYPOINT_COUNT):
        if index >= len(hand):
            values.extend([0.0, 0.0, 0.0])
        else:
            values.extend(_to_2d_values(hand[index], image_width, image_height))
    return values


def _hand_to_openpose_3d(hand: list[Landmark]) -> list[float]:
    if not hand:
        return _zero_keypoints_3d(HAND_KEYPOINT_COUNT)

    values: list[float] = []
    for index in range(HAND_KEYPOINT_COUNT):
        if index >= len(hand):
            values.extend([0.0, 0.0, 0.0, 0.0])
        else:
            values.extend(_to_3d_values(hand[index]))
    return values


def _coerce_keypoints(keypoints: ExtractedKeypoints | dict[str, Any]):
    if isinstance(keypoints, ExtractedKeypoints):
        return keypoints
    return ExtractedKeypoints.model_validate(keypoints)


def convert_to_openpose(
    keypoints: ExtractedKeypoints | dict[str, Any],
) -> OpenPoseResult:
    extracted_keypoints = _coerce_keypoints(keypoints)

    return OpenPoseResult(
        people=OpenPosePerson(
            face_keypoints_2d=_zero_keypoints_2d(FACE_KEYPOINT_COUNT),
            pose_keypoints_2d=_pose_to_body25_2d(extracted_keypoints),
            hand_left_keypoints_2d=_hand_to_openpose_2d(
                extracted_keypoints.left_hand,
                extracted_keypoints.image_width,
                extracted_keypoints.image_height,
            ),
            hand_right_keypoints_2d=_hand_to_openpose_2d(
                extracted_keypoints.right_hand,
                extracted_keypoints.image_width,
                extracted_keypoints.image_height,
            ),
            face_keypoints_3d=_zero_keypoints_3d(FACE_KEYPOINT_COUNT),
            pose_keypoints_3d=_pose_to_body25_3d(extracted_keypoints),
            hand_left_keypoints_3d=_hand_to_openpose_3d(
                extracted_keypoints.left_hand
            ),
            hand_right_keypoints_3d=_hand_to_openpose_3d(
                extracted_keypoints.right_hand
            ),
        )
    )
