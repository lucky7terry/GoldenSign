"""모델 파일이 없을 때와 반복 호출 시의 동작을 고정한다.

TensorFlow 없이 도는 경로만 다룬다. load_recognition_model 은 파일 존재를
먼저 확인하고 그 뒤에야 keras 를 임포트하므로, 파일이 없는 경로는 CI 에서
검증할 수 있다.
"""

import os
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.recognition_model import (  # noqa: E402
    FEATURE_DIM,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
    RecognitionModelUnavailableError,
    load_recognition_model,
    load_recognition_models,
    model_path,
    model_paths,
)
from unittest import mock  # noqa: E402


class RecognitionModelTest(unittest.TestCase):
    def test_contract_constants_match_the_trained_model(self):
        # 학습 노트북의 final_summary.json 과 같아야 한다.
        self.assertEqual(SEQUENCE_LENGTH, 60)
        self.assertEqual(FEATURE_DIM, 420)
        self.assertEqual(NUM_CLASSES, 50)

    def test_missing_file_raises_a_clear_error(self):
        with self.assertRaises(RecognitionModelUnavailableError) as caught:
            load_recognition_model(Path("models/definitely-not-here.keras"))

        message = str(caught.exception)
        self.assertIn("definitely-not-here.keras", message)
        # 파일이 커밋되지 않는다는 걸 메시지가 알려줘야 한다.
        self.assertIn("not committed", message)

    def test_model_path_points_into_the_models_directory(self):
        self.assertEqual(model_path().parent.name, "models")

    def test_model_filename_can_be_overridden(self):
        previous = os.environ.get("RECOGNITION_MODEL_FILENAME")
        os.environ["RECOGNITION_MODEL_FILENAME"] = "model_fold3.keras"
        try:
            self.assertEqual(model_path().name, "model_fold3.keras")
        finally:
            if previous is None:
                del os.environ["RECOGNITION_MODEL_FILENAME"]
            else:
                os.environ["RECOGNITION_MODEL_FILENAME"] = previous



_ENV = ("RECOGNITION_MODEL_FILENAMES", "RECOGNITION_MODEL_FILENAME")


def _env(**values):
    """두 환경변수를 지운 뒤 주어진 값만 넣는다."""
    base = {k: v for k, v in os.environ.items() if k not in _ENV}
    base.update(values)
    return mock.patch.dict(os.environ, base, clear=True)


class EnsembleModelPathsTest(unittest.TestCase):
    """서버는 5-fold 모델을 전부 올려 확률을 평균한다."""

    def test_default_is_the_five_fold_models(self):
        with _env():
            names = [p.name for p in model_paths()]
        self.assertEqual(names, [f"model_fold{k}.keras" for k in range(5)])

    def test_the_list_can_be_overridden(self):
        with _env(RECOGNITION_MODEL_FILENAMES="model_fold0.keras, model_fold2.keras"):
            names = [p.name for p in model_paths()]
        self.assertEqual(names, ["model_fold0.keras", "model_fold2.keras"])

    def test_the_old_single_file_setting_still_means_one_model(self):
        """이미 한 파일로 배포한 곳이 조용히 앙상블로 바뀌면 안 된다."""
        with _env(RECOGNITION_MODEL_FILENAME="model_fold3.keras"):
            names = [p.name for p in model_paths()]
        self.assertEqual(names, ["model_fold3.keras"])

    def test_an_empty_list_is_rejected(self):
        with _env(RECOGNITION_MODEL_FILENAMES=" , "):
            with self.assertRaises(RecognitionModelUnavailableError):
                model_paths()

    def test_a_duplicate_is_rejected(self):
        """평균에서 그 모델만 두 배 무게를 받는데 겉으로는 정상이다."""
        with _env(RECOGNITION_MODEL_FILENAMES="model_fold0.keras,model_fold0.keras"):
            with self.assertRaises(RecognitionModelUnavailableError):
                model_paths()

    def test_one_missing_file_fails_the_whole_ensemble(self):
        """일부만 올라온 채로 3개, 4개 평균을 내면 아무도 모른다."""
        with _env(RECOGNITION_MODEL_FILENAMES="model_fold0.keras,definitely-not-here.keras"):
            with mock.patch(
                "app.services.recognition_model.load_recognition_model",
                side_effect=lambda path: (
                    object() if path.name != "definitely-not-here.keras"
                    else load_recognition_model(Path("models/definitely-not-here.keras"))
                ),
            ):
                with self.assertRaises(RecognitionModelUnavailableError):
                    load_recognition_models()


if __name__ == "__main__":
    unittest.main()
