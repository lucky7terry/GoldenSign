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
    model_path,
)


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


if __name__ == "__main__":
    unittest.main()
