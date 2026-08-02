from pydantic import BaseModel, Field


class CameraIntrinsics(BaseModel):
    data: str = ""


class CameraMatrix(BaseModel):
    data: str = ""


class CameraDistortion(BaseModel):
    rows: str = "0"
    data: str = ""


class CameraParameters(BaseModel):
    intrinsics: CameraIntrinsics = Field(
        default_factory=CameraIntrinsics,
        alias="Intrinsics",
    )
    camera_matrix: CameraMatrix = Field(
        default_factory=CameraMatrix,
        alias="CameraMatrix",
    )
    distortion: CameraDistortion = Field(
        default_factory=CameraDistortion,
        alias="Distortion",
    )


class OpenPosePerson(BaseModel):
    person_id: int = -1
    face_keypoints_2d: list[float] = Field(default_factory=list)
    pose_keypoints_2d: list[float] = Field(default_factory=list)
    hand_left_keypoints_2d: list[float] = Field(default_factory=list)
    hand_right_keypoints_2d: list[float] = Field(default_factory=list)
    face_keypoints_3d: list[float] = Field(default_factory=list)
    pose_keypoints_3d: list[float] = Field(default_factory=list)
    hand_left_keypoints_3d: list[float] = Field(default_factory=list)
    hand_right_keypoints_3d: list[float] = Field(default_factory=list)


class OpenPoseResult(BaseModel):
    version: float = 1.3
    people: OpenPosePerson = Field(default_factory=OpenPosePerson)
    camparam: CameraParameters = Field(default_factory=CameraParameters)
