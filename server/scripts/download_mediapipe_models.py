from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve


MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/"
        "pose_landmarker_lite.task"
    ),
}


def main() -> None:
    server_directory = Path(__file__).resolve().parents[1]
    models_directory = server_directory / "models"
    models_directory.mkdir(exist_ok=True)

    for filename, url in MODELS.items():
        target = models_directory / filename
        if target.exists():
            print(f"Already exists: {target}")
            continue

        print(f"Downloading {filename}...")
        with NamedTemporaryFile(
            delete=False,
            dir=models_directory,
            suffix=".tmp",
        ) as temporary_file:
            temporary_target = Path(temporary_file.name)

        try:
            urlretrieve(url, temporary_target)
            temporary_target.replace(target)
        except Exception:
            temporary_target.unlink(missing_ok=True)
            raise
        print(f"Saved: {target}")


if __name__ == "__main__":
    main()
