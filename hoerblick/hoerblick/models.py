from __future__ import annotations

import hashlib
import csv
import io
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .model_download import resolve_own_model
from .priority import CLASS_SPECS, normalize_label


try:
    import tensorflow as _tf

    @_tf.keras.utils.register_keras_serializable(package="hoerblick")
    class ForestInference(_tf.keras.layers.Layer):
        def __init__(self, trees: list[dict], **kwargs):
            super().__init__(**kwargs)
            self.trees = trees

        def call(self, inputs):
            tree_outputs = []
            for tree in self.trees:
                left = _tf.constant(tree["children_left"], dtype=_tf.int32)
                right = _tf.constant(tree["children_right"], dtype=_tf.int32)
                features = _tf.constant(tree["features"], dtype=_tf.int32)
                thresholds = _tf.constant(tree["thresholds"], dtype=_tf.float32)
                values = _tf.constant(tree["values"], dtype=_tf.float32)
                nodes = _tf.zeros([_tf.shape(inputs)[0]], dtype=_tf.int32)
                for _ in range(tree["max_depth"] + 1):
                    node_left = _tf.gather(left, nodes)
                    node_right = _tf.gather(right, nodes)
                    is_leaf = _tf.equal(node_left, node_right)
                    feature = _tf.maximum(_tf.gather(features, nodes), 0)
                    feature_value = _tf.gather(inputs, feature, axis=1, batch_dims=1)
                    threshold = _tf.gather(thresholds, nodes)
                    next_node = _tf.where(feature_value <= threshold, node_left, node_right)
                    nodes = _tf.where(is_leaf, nodes, next_node)
                leaf_values = _tf.gather(values, nodes)
                probabilities = leaf_values / _tf.maximum(
                    _tf.reduce_sum(leaf_values, axis=1, keepdims=True), 1e-8
                )
                tree_outputs.append(probabilities)
            return _tf.reduce_mean(_tf.stack(tree_outputs, axis=0), axis=0)

        def get_config(self):
            return {**super().get_config(), "trees": self.trees}

    @_tf.keras.utils.register_keras_serializable(package="hoerblick")
    class RBFExpansion(_tf.keras.layers.Layer):
        def __init__(self, centers: list[list[float]], gamma: float, **kwargs):
            super().__init__(**kwargs)
            self.centers = centers
            self.gamma = float(gamma)

        def build(self, input_shape):
            self.center_values = self.add_weight(
                name="centers",
                shape=(len(self.centers), len(self.centers[0])),
                initializer=_tf.keras.initializers.Constant(self.centers),
                trainable=False,
            )
            super().build(input_shape)

        def call(self, inputs):
            differences = _tf.expand_dims(inputs, axis=1) - _tf.expand_dims(
                self.center_values, axis=0
            )
            squared_distances = _tf.reduce_sum(_tf.square(differences), axis=-1)
            return _tf.exp(-self.gamma * squared_distances)

        def get_config(self):
            return {
                **super().get_config(),
                "centers": self.centers,
                "gamma": self.gamma,
            }

    @_tf.keras.utils.register_keras_serializable(package="hoerblick")
    class PairwiseSVCInference(_tf.keras.layers.Layer):
        def __init__(
            self,
            support_counts: list[int],
            dual_coefficients: list[list[float]],
            intercepts: list[float],
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.support_counts = [int(value) for value in support_counts]
            self.initial_dual_coefficients = dual_coefficients
            self.initial_intercepts = intercepts

        def build(self, input_shape):
            self.dual_coefficients = self.add_weight(
                name="dual_coefficients",
                shape=(
                    len(self.initial_dual_coefficients),
                    len(self.initial_dual_coefficients[0]),
                ),
                initializer=_tf.keras.initializers.Constant(
                    self.initial_dual_coefficients
                ),
                trainable=True,
            )
            self.intercepts = self.add_weight(
                name="intercepts",
                shape=(len(self.initial_intercepts),),
                initializer=_tf.keras.initializers.Constant(self.initial_intercepts),
                trainable=True,
            )
            super().build(input_shape)

        def call(self, kernel_values):
            class_count = len(self.support_counts)
            starts = [0]
            for count in self.support_counts:
                starts.append(starts[-1] + count)
            votes = _tf.zeros(
                [_tf.shape(kernel_values)[0], class_count],
                dtype=kernel_values.dtype,
            )
            margins = _tf.zeros_like(votes)
            pair_index = 0
            for first_class in range(class_count):
                for second_class in range(first_class + 1, class_count):
                    first_start, first_end = starts[first_class : first_class + 2]
                    second_start, second_end = starts[second_class : second_class + 2]
                    decision = (
                        _tf.reduce_sum(
                            kernel_values[:, first_start:first_end]
                            * self.dual_coefficients[
                                second_class - 1, first_start:first_end
                            ],
                            axis=1,
                        )
                        + _tf.reduce_sum(
                            kernel_values[:, second_start:second_end]
                            * self.dual_coefficients[
                                first_class, second_start:second_end
                            ],
                            axis=1,
                        )
                        + self.intercepts[pair_index]
                    )
                    first_wins = _tf.cast(
                        decision > 0, dtype=kernel_values.dtype
                    )[:, _tf.newaxis]
                    first_hot = _tf.one_hot(
                        first_class, class_count, dtype=kernel_values.dtype
                    )[_tf.newaxis, :]
                    second_hot = _tf.one_hot(
                        second_class, class_count, dtype=kernel_values.dtype
                    )[_tf.newaxis, :]
                    votes += first_wins * first_hot + (1 - first_wins) * second_hot
                    margins += decision[:, _tf.newaxis] * (first_hot - second_hot)
                    pair_index += 1
            logits = votes + 0.1 * _tf.math.tanh(margins / class_count)
            return _tf.nn.softmax(logits, axis=-1)

        def get_config(self):
            return {
                **super().get_config(),
                "support_counts": self.support_counts,
                "dual_coefficients": self.initial_dual_coefficients,
                "intercepts": self.initial_intercepts,
            }
except ImportError:
    _tf = None
    ForestInference = None
    RBFExpansion = None
    PairwiseSVCInference = None


class ModelUnavailable(RuntimeError):
    """Raised when a real model cannot be loaded in the current environment."""


@dataclass(frozen=True)
class ModelResult:
    label: str
    confidence: float
    model_name: str
    is_demo: bool = False
    source_label: str = ""


SAMPLE_RATE = 16_000
CLIP_SECONDS = 5
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
# Exact names from the class map bundled with YAMNet / AudioSet.
YAMNET_CLASSES = {
    "Smoke detector, smoke alarm": "alarm",
    "Fire alarm": "alarm",
    "Siren": "siren",
    "Civil defense siren": "siren",
    "Emergency vehicle": "siren",
    "Police car (siren)": "siren",
    "Ambulance (siren)": "siren",
    "Fire engine, fire truck (siren)": "siren",
    "Doorbell": "doorbell",
    "Ding-dong": "doorbell",
}


def decode_audio(audio_bytes: bytes):
    """Both models see the same first five seconds, mono float32 at 16 kHz."""
    import librosa
    import numpy as np
    import soundfile as sf

    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as source:
            rate = source.samplerate
            audio = source.read(rate * CLIP_SECONDS, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError("Leere oder ungültige Audiodaten.")
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
        return np.clip(audio, -1, 1).astype(np.float32)
    except (ValueError, RuntimeError, OSError) as exc:
        raise ValueError("Audio nicht lesbar. Bitte eine gültige WAV-, MP3-, FLAC- oder OGG-Datei verwenden.") from exc


@lru_cache(maxsize=4)
def _load_own_model(model_path: str, modified_ns: int):
    del modified_ns  # Cache invalidation when the export changes.
    if _tf is None:
        raise ModelUnavailable("TensorFlow fehlt. Bitte requirements-ml.txt installieren.")
    model = _tf.keras.models.load_model(
        model_path, compile=False,
        custom_objects={"ForestInference": ForestInference, "RBFExpansion": RBFExpansion,
                        "PairwiseSVCInference": PairwiseSVCInference},
    )
    if tuple(model.input_shape) != (None, 448) or tuple(model.output_shape) != (None, 4):
        raise ModelUnavailable("Dieser Adapter erwartet den vorhandenen 448-Merkmale-/4-Klassen-Export.")
    return model


@lru_cache(maxsize=2)
def _load_yamnet(project_root: str):
    import tensorflow_hub as hub

    cache_dir = Path(project_root) / "models" / "model_b" / "tfhub_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TFHUB_CACHE_DIR", str(cache_dir))
    model = hub.load(YAMNET_HANDLE)
    with open(model.class_map_path().numpy().decode("utf-8"), encoding="utf-8") as source:
        labels = [row["display_name"] for row in csv.DictReader(source)]
    if len(labels) != 521 or not set(YAMNET_CLASSES).issubset(labels):
        raise ModelUnavailable("YAMNet-Klassenliste passt nicht zum dokumentierten Mapping.")
    return model, labels


def map_yamnet_scores(scores, labels: list[str]) -> ModelResult:
    """Mean across frames, then top AudioSet class; never sum sigmoid scores."""
    import numpy as np

    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != len(labels) or not np.isfinite(scores).all():
        raise ModelUnavailable("Ungültige YAMNet-Ausgabe.")
    means = scores.mean(axis=0)
    index = int(means.argmax())
    source_label = labels[index]
    return ModelResult(
        label=YAMNET_CLASSES.get(source_label, "background"),
        confidence=float(means[index]), model_name="YAMNet", source_label=source_label,
    )


def _filename_hint(filename: str) -> str | None:
    name = filename.lower()
    patterns = {
        "alarm": r"alarm|rauch|feuer",
        "siren": r"siren|martin|warn",
        "doorbell": r"door|klingel|bell",
        "background": r"background|hinter|silence|stille",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, name):
            return label
    return None


class DemoAdapter:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def predict(self, audio_bytes: bytes, filename: str) -> ModelResult:
        hint = _filename_hint(filename)
        if hint is None:
            labels = list(CLASS_SPECS)
            digest = hashlib.sha256(audio_bytes or filename.encode("utf-8")).digest()
            hint = labels[digest[0] % len(labels)]
        confidence = 0.91 if _filename_hint(filename) else 0.58
        return ModelResult(
            label=normalize_label(hint),
            confidence=confidence,
            model_name=f"{self.model_name} (Demo-Modus)",
            is_demo=True,
        )


class OwnModelAdapter:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.model_path = project_root / "models" / "model_a" / "keras_model.h5"
        self.labels_path = project_root / "models" / "model_a" / "labels.txt"

    def load(self):
        labels = _read_labels(self.labels_path)
        if labels != ["alarm", "siren", "doorbell", "background"]:
            raise ModelUnavailable("labels.txt muss alarm, siren, doorbell, background in dieser Reihenfolge enthalten.")
        try:
            model_path = resolve_own_model(self.model_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ModelUnavailable(f"Eigenes Modell konnte nicht heruntergeladen werden: {exc}") from exc
        return _load_own_model(str(model_path.resolve()), model_path.stat().st_mtime_ns), labels

    def predict(self, audio_bytes: bytes, filename: str) -> ModelResult:
        model, labels = self.load()
        audio = decode_audio(audio_bytes)
        model_input = _prepare_audio_input(_tf, audio, model.input_shape)
        prediction = model(model_input, training=False).numpy()
        scores = prediction[0] if getattr(prediction, "ndim", 1) > 1 else prediction
        index = int(scores.argmax())
        if index >= len(labels):
            raise ModelUnavailable(
                "Das Modell liefert mehr Klassen als labels.txt enthält."
            )
        confidence = float(scores[index])
        return ModelResult(
            label=normalize_label(labels[index]),
            confidence=max(0.0, min(1.0, confidence)),
            model_name="Eigenes Modell",
        )


class YamnetAdapter:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def load(self):
        try:
            return _load_yamnet(str(self.project_root.resolve()))
        except Exception as exc:
            raise ModelUnavailable(f"YAMNet konnte nicht geladen werden: {exc}") from exc

    def predict(self, audio_bytes: bytes, filename: str) -> ModelResult:
        import numpy as np

        model, labels = self.load()
        audio = decode_audio(audio_bytes)
        # YAMNet needs at least one 0.975-second waveform window.
        if len(audio) < 15_600:
            audio = np.pad(audio, (0, 15_600 - len(audio)))
        scores, _embeddings, _spectrogram = model(_tf.convert_to_tensor(audio))
        return map_yamnet_scores(scores.numpy(), labels)


def get_model_adapter(model_key: str, project_root: Path):
    if model_key == "demo":
        return DemoAdapter("Demo")
    if model_key == "own":
        return OwnModelAdapter(project_root)
    if model_key == "yamnet":
        return YamnetAdapter(project_root)
    raise ValueError(f"Unbekannter Modellschlüssel: {model_key}")


def model_status(model_key: str, project_root: Path | None = None) -> dict[str, bool | str]:
    root = project_root or Path.cwd()
    try:
        get_model_adapter(model_key, root).load()
        return {"ready": True, "message": "Modell geladen · lokale Analyse"}
    except Exception as exc:
        return {"ready": False, "message": str(exc)}


def _read_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        return []
    labels = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        value = re.sub(r"^\s*\d+\s+", "", line).strip()
        if value:
            labels.append(value)
    return labels


def _prepare_audio_input(tf, audio, input_shape):
    """Create a robust tensor for common Teachable-Machine audio exports."""
    shape = tuple(dim if dim is not None else 0 for dim in input_shape)
    if len(shape) == 2 and shape[1] == 448:
        import numpy as np

        target_samples = 5 * 16_000
        waveform = tf.convert_to_tensor(audio[:target_samples], dtype=tf.float32)
        if int(waveform.shape[0]) < target_samples:
            waveform = tf.pad(waveform, [[0, target_samples - int(waveform.shape[0])]])
        stft = tf.signal.stft(
            waveform, frame_length=400, frame_step=160, fft_length=512
        )
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=64,
            num_spectrogram_bins=257,
            sample_rate=16_000,
            lower_edge_hertz=80.0,
            upper_edge_hertz=7_600.0,
        )
        mel = tf.math.log(tf.matmul(tf.abs(stft), mel_matrix) + 1e-6).numpy()
        delta = np.diff(mel, axis=0)
        features = np.concatenate([
            mel.mean(axis=0),
            mel.std(axis=0),
            np.quantile(mel, [0.1, 0.5, 0.9], axis=0).ravel(),
            delta.mean(axis=0),
            delta.std(axis=0),
        ]).astype(np.float32)
        return tf.convert_to_tensor(features[np.newaxis, :], dtype=tf.float32)

    if len(shape) == 2 and shape[1] > 0:
        target_length = shape[1]
        waveform = audio[:target_length]
        if waveform.size < target_length:
            waveform = tf.pad(waveform, [[0, target_length - waveform.size]])
        return tf.expand_dims(waveform, axis=0)

    if len(shape) not in {3, 4}:
        raise ModelUnavailable(
            f"Nicht unterstützte Modell-Eingabeform: {input_shape}."
        )

    target_height = shape[1] or 498
    target_width = shape[2] or 64
    target_samples = 5 * 16_000
    waveform = tf.convert_to_tensor(audio[:target_samples], dtype=tf.float32)
    if int(waveform.shape[0]) < target_samples:
        waveform = tf.pad(waveform, [[0, target_samples - int(waveform.shape[0])]])
    spectrogram = tf.abs(
        tf.signal.stft(waveform, frame_length=400, frame_step=160, fft_length=512)
    )
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=257,
        sample_rate=16_000,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7_600.0,
    )
    spectrogram = tf.math.log(tf.matmul(spectrogram, mel_matrix) + 1e-6)
    if target_height == 498 and target_width == 64:
        return tf.expand_dims(spectrogram, axis=0)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    if (target_height, target_width) != (498, 64):
        spectrogram = tf.image.resize(spectrogram, [target_height, target_width])
    if len(shape) == 3:
        spectrogram = tf.squeeze(spectrogram, axis=-1)
        return tf.expand_dims(spectrogram, axis=0)

    spectrogram = tf.expand_dims(spectrogram, axis=0)
    if len(shape) == 4:
        channels = shape[3] or 1
        if channels > 1:
            spectrogram = tf.repeat(spectrogram, channels, axis=-1)
    return spectrogram
