from pydantic import BaseModel, Field


class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: float | None = None


class ExtractedKeypoints(BaseModel):
    face: list[Landmark] = Field(default_factory=list)
    pose: list[Landmark] = Field(default_factory=list)
    left_hand: list[Landmark] = Field(default_factory=list)
    right_hand: list[Landmark] = Field(default_factory=list)

    image_width: int | None = None
    image_height: int | None = None

    face_detected: bool = False
    pose_detected: bool = False
    left_hand_detected: bool = False
    right_hand_detected: bool = False


KeypointResult = ExtractedKeypoints
