"""수어 단어 인식 모델 로딩.

학습 노트북(transformer_tuning_얼굴포함_개선.ipynb)이 저장한 Keras 모델을
읽는다. 모델은 (60, 420) 특징 시퀀스를 받아 50개 단어에 대한 확률을 낸다.

커스텀 레이어 3개가 `sign>` 패키지 이름으로 직렬화되어 있어, 같은 이름으로
등록된 정의가 프로세스 안에 있어야 load_model 이 성공한다. 아래 클래스들은
학습 코드의 정의를 그대로 옮긴 것이다 — 추론에만 필요한 경로만 남겼다.
"""

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 60
FEATURE_DIM = 420
NUM_CLASSES = 50

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
_DEFAULT_MODEL_FILENAME = "model_fold0.keras"


class RecognitionModelUnavailableError(RuntimeError):
    """인식 모델을 로드하지 못한 상태.

    모델 파일이 없거나 TensorFlow 를 쓸 수 없는 경우다. 프레임마다 다시
    시도해도 결과가 같으므로 재시도 대상이 아니다.
    """


def _register_custom_layers():
    """학습 때와 동일한 이름으로 커스텀 레이어를 등록한다.

    임포트 시점이 아니라 호출 시점에 keras 를 들여온다. TensorFlow 는 무겁고,
    모델을 쓰지 않는 실행 경로(테스트 등)까지 끌고 들어갈 이유가 없다.
    """
    import keras
    import tensorflow as tf
    from keras import layers

    @keras.utils.register_keras_serializable(package="sign")
    class PositionalEmbedding(layers.Layer):
        def __init__(self, seq_len, d_model, **kwargs):
            super().__init__(**kwargs)
            self.seq_len, self.d_model = seq_len, d_model
            self.emb = layers.Embedding(seq_len, d_model)

        def call(self, x):
            return x + self.emb(tf.range(self.seq_len))[None]

        def get_config(self):
            return {**super().get_config(),
                    "seq_len": self.seq_len, "d_model": self.d_model}

    @keras.utils.register_keras_serializable(package="sign")
    class DropPath(layers.Layer):
        """학습 전용 정규화. 추론에서는 입력을 그대로 흘린다."""

        def __init__(self, rate=0.0, **kwargs):
            super().__init__(**kwargs)
            self.rate = float(rate)

        def call(self, x, training=None):
            if (not training) or self.rate == 0.0:
                return x
            keep = 1.0 - self.rate
            shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
            return x / keep * tf.floor(keep + tf.random.uniform(shape, dtype=x.dtype))

        def get_config(self):
            return {**super().get_config(), "rate": self.rate}

    @keras.utils.register_keras_serializable(package="sign")
    class AttentionPool(layers.Layer):
        """프레임별 중요도를 학습해 가중 평균한다."""

        def build(self, input_shape):
            self.score = layers.Dense(1)
            super().build(input_shape)

        def call(self, x):
            w = tf.nn.softmax(self.score(x), axis=1)
            return tf.reduce_sum(w * x, axis=1)

    return {
        "sign>PositionalEmbedding": PositionalEmbedding,
        "sign>DropPath": DropPath,
        "sign>AttentionPool": AttentionPool,
    }


def model_path() -> Path:
    return _MODEL_DIR / os.getenv("RECOGNITION_MODEL_FILENAME", _DEFAULT_MODEL_FILENAME)


def load_recognition_model(path: Path | None = None):
    """모델을 읽어 돌려준다. 실패하면 RecognitionModelUnavailableError."""
    target = path or model_path()
    if not target.exists():
        raise RecognitionModelUnavailableError(
            f"Recognition model not found: {target}. "
            "Place the .keras file there (it is not committed to the repository)."
        )

    try:
        custom_objects = _register_custom_layers()
        import keras

        model = keras.models.load_model(target, custom_objects=custom_objects)
    except RecognitionModelUnavailableError:
        raise
    except Exception as exc:
        raise RecognitionModelUnavailableError(
            f"Failed to load recognition model at {target}: {exc}"
        ) from exc

    expected = (None, SEQUENCE_LENGTH, FEATURE_DIM)
    actual = tuple(model.input_shape)
    if actual != expected:
        raise RecognitionModelUnavailableError(
            f"Model input shape {actual} does not match the server contract {expected}."
        )
    if model.output_shape[-1] != NUM_CLASSES:
        raise RecognitionModelUnavailableError(
            f"Model outputs {model.output_shape[-1]} classes, expected {NUM_CLASSES}."
        )

    logger.info(
        "Recognition model loaded",
        extra={"path": str(target), "input_shape": str(actual)},
    )
    return model


def make_predictor(model, batch_size: int = 1):
    """추론 함수를 고정 시그니처 tf.function 으로 감싼다.

    호출 방식만으로 12배 차이가 난다(이 저장소 기준 실측, 배치 1):

        model.predict(x)           54.9 ms
        model(x, training=False)   36.9 ms
        tf.function 고정 시그니처    4.4 ms

    predict() 는 호출마다 데이터 어댑터와 콜백 경로를 새로 세우고,
    model(x) 도 입력 모양이 바뀔 수 있다고 보고 매번 확인한다. 모양이
    (batch, 60, 420) 으로 고정이라는 걸 알려주면 그래프를 한 번만 만든다.

    실시간 경로에서는 이 함수가 돌려주는 것을 쓰고 predict() 를 부르지 말 것.
    """
    import tensorflow as tf

    @tf.function(
        input_signature=[
            tf.TensorSpec([batch_size, SEQUENCE_LENGTH, FEATURE_DIM], tf.float32)
        ]
    )
    def _infer(features):
        return model(features, training=False)

    return _infer


_model = None
_initialization_error: RecognitionModelUnavailableError | None = None
_model_lock = threading.Lock()


_predictor = None
_predictor_lock = threading.Lock()


def get_recognition_model():
    """모델을 한 번만 읽어 재사용한다. 실패했으면 같은 예외를 즉시 돌려준다.

    로딩에 수 초가 걸린다(실측 3~8초). 캐싱하지 않으면 부를 때마다 그만큼
    멈추고, 실패를 기억하지 않으면 단어마다 로딩을 재시도하게 된다.
    MediaPipe 쪽에서 같은 실수로 프레임마다 재시도가 돌았던 적이 있다.
    """
    global _model, _initialization_error

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        if _initialization_error is not None:
            raise _initialization_error
        try:
            _model = load_recognition_model()
        except RecognitionModelUnavailableError as exc:
            _initialization_error = exc
            raise
        except Exception as exc:
            _initialization_error = RecognitionModelUnavailableError(str(exc))
            raise _initialization_error from exc

    return _model


def preload_recognition_model() -> bool:
    """기동 시 모델을 미리 올린다. 실패해도 예외를 밖으로 내지 않는다.

    아직 lifespan 에 연결하지 않았다. 추론을 실제로 돌리는 단계에서 붙인다 —
    지금 붙이면 쓰지도 않는 모델 때문에 기동이 수 초 느려진다.
    """
    try:
        get_recognition_model()
    except RecognitionModelUnavailableError as exc:
        logger.error(
            "Recognition model unavailable; word recognition is disabled",
            extra={"error": str(exc)},
        )
        return False
    return True


def get_recognition_predictor():
    """추론 함수. 모델과 마찬가지로 한 번만 만든다.

    make_predictor 는 호출할 때마다 새 tf.function 과 그래프를 만든다.
    단어마다 부르면 그래프가 쌓이고 첫 호출마다 트레이싱 비용을 다시 낸다.
    """
    global _predictor
    if _predictor is not None:
        return _predictor
    with _predictor_lock:
        if _predictor is None:
            _predictor = make_predictor(get_recognition_model())
    return _predictor


def recognition_model_available() -> bool:
    return _model is not None


def recognition_model_error() -> str | None:
    if _initialization_error is None:
        return None
    return str(_initialization_error)
