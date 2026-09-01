"""손 좌우 배정.

MediaPipe 손 검출기의 handedness 라벨은 셀피(거울) 화면을 전제한다.
정면을 향하는 안경 카메라에서는 좌우가 뒤바뀌므로, 신체 구조로 추정되어
이 문제가 없는 Pose 모델의 손목 좌표를 기준으로 배정한다.

MediaPipe에 의존하지 않는 순수 기하 로직이라 단독으로 테스트할 수 있다.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# MediaPipe Pose 랜드마크 인덱스
POSE_LEFT_WRIST_INDEX = 15
POSE_RIGHT_WRIST_INDEX = 16

MIN_WRIST_VISIBILITY = 0.5

Point = tuple[float, float]
HandCandidate = dict[str, Any]
Assignment = dict[str, HandCandidate | None]


def _pose_wrist_point(
    pose: list[dict[str, float | None]],
    index: int,
) -> Point | None:
    if index >= len(pose):
        return None
    landmark = pose[index]
    visibility = landmark.get("visibility")
    if visibility is None or visibility < MIN_WRIST_VISIBILITY:
        return None
    return (float(landmark["x"]), float(landmark["y"]))


def _hand_wrist_point(candidate: HandCandidate) -> Point | None:
    landmarks = candidate.get("landmarks")
    if not landmarks:
        return None
    return (float(landmarks[0]["x"]), float(landmarks[0]["y"]))


def _squared_distance(first: Point, second: Point) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _score_or_default(score: float | None) -> float:
    if score is None:
        return -1.0
    return score


def assign_by_pose_wrists(
    hand_candidates: list[HandCandidate],
    pose: list[dict[str, float | None]],
) -> Assignment | None:
    """Pose 손목에 가장 가까운 손을 좌/우로 배정한다.

    Pose 손목을 쓸 수 없으면 None을 돌려준다(호출부가 폴백).
    """
    left_wrist = _pose_wrist_point(pose, POSE_LEFT_WRIST_INDEX)
    right_wrist = _pose_wrist_point(pose, POSE_RIGHT_WRIST_INDEX)
    if left_wrist is None and right_wrist is None:
        return None

    usable = [
        candidate
        for candidate in hand_candidates
        if _hand_wrist_point(candidate) is not None
    ][:2]
    if not usable:
        return {"left": None, "right": None}

    if len(usable) == 1:
        candidate = usable[0]
        point = _hand_wrist_point(candidate)
        to_left = (
            _squared_distance(point, left_wrist)
            if left_wrist is not None
            else float("inf")
        )
        to_right = (
            _squared_distance(point, right_wrist)
            if right_wrist is not None
            else float("inf")
        )
        if to_left <= to_right:
            return {"left": candidate, "right": None}
        return {"left": None, "right": candidate}

    first, second = usable
    first_point = _hand_wrist_point(first)
    second_point = _hand_wrist_point(second)

    def pairing_cost(left_point: Point, right_point: Point) -> float:
        cost = 0.0
        if left_wrist is not None:
            cost += _squared_distance(left_point, left_wrist)
        if right_wrist is not None:
            cost += _squared_distance(right_point, right_wrist)
        return cost

    straight = pairing_cost(first_point, second_point)
    swapped = pairing_cost(second_point, first_point)
    if straight <= swapped:
        return {"left": first, "right": second}
    return {"left": second, "right": first}


def assign_by_handedness(hand_candidates: list[HandCandidate]) -> Assignment:
    """폴백. 검출기 handedness 라벨을 그대로 신뢰한다."""
    assignment: Assignment = {"left": None, "right": None}
    selected_indexes: set[int] = set()

    for label in ("left", "right"):
        labeled = [
            candidate
            for candidate in hand_candidates
            if candidate["label"] == label
        ]
        if not labeled:
            continue
        if len(labeled) > 1:
            logger.warning(
                "Duplicate %s hand detected; preserving higher score",
                label,
            )
        selected = max(
            labeled,
            key=lambda candidate: _score_or_default(candidate["score"]),
        )
        selected_indexes.add(selected["index"])
        assignment[label] = selected

    for label in ("left", "right"):
        if assignment[label] is not None:
            continue
        remaining = [
            candidate
            for candidate in hand_candidates
            if candidate["index"] not in selected_indexes
        ]
        if not remaining:
            continue
        selected = max(
            remaining,
            key=lambda candidate: _score_or_default(candidate["score"]),
        )
        selected_indexes.add(selected["index"])
        logger.warning(
            "Reassigned duplicate or unlabeled hand to missing %s hand",
            label,
        )
        assignment[label] = selected

    return assignment


def assign_hands(
    hand_candidates: list[HandCandidate],
    pose: list[dict[str, float | None]],
) -> Assignment:
    """Pose 손목 기준으로 배정하고, 불가능하면 handedness로 폴백한다."""
    assignment = assign_by_pose_wrists(hand_candidates, pose)
    if assignment is not None:
        return assignment

    if hand_candidates:
        logger.warning(
            "Pose wrists unavailable; falling back to MediaPipe handedness",
            extra={"hand_count": len(hand_candidates)},
        )
    return assign_by_handedness(hand_candidates)
