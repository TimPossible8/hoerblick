from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .priority import CLASS_SPECS, normalize_label


class ModelUnavailable(RuntimeError):
    """Raised when a real model cannot be loaded in the current environment."""


@dataclass(frozen=True)
class ModelResult:
    label: str
    confidence: float
    model_name: str
    is_demo: bool = False


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

    def predict(self, audio_bytes: bytes, filename: str) -> ModelResult:
        try:
            import tensorflow as tf
            import librosa
        except ImportError as exc:
            raise ModelUnavailable(
                "TensorFlow und librosa sind nicht vollständig installiert. "
                "Installiere requirements-ml.txt."
            ) from exc
        if not self.model_path.exists():
            raise ModelUnavailable(
                f"Die exportierte Modell-Datei fehlt: {self.model_path}. "
                "Exportiere das Teachable-Machine-Modell nach models/model_a/."
            )
        labels = _read_labels(self.labels_path)
        if not labels:
            raise ModelUnavailable(
                f"Die Label-Datei fehlt oder ist leer: {self.labels_path}."
            )

        try:
            audio, _ = librosa.load(
                librosa.io.BytesIO(audio_bytes),
                sr=16000,
                mono=True,
            )
        except AttributeError:
            import io

            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        if audio.size == 0:
            raise ModelUnavailable(f"Die Audiodatei ist leer: {filename}.")

        model = tf.keras.models.load_model(self.model_path, compile=False)
        model_input = _prepare_audio_input(tf, audio, model.input_shape)
        prediction = model.predict(model_input, verbose=0)
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

    def predict(self, audio_bytes: bytes, filename: str) -> ModelResult:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as exc:
            raise ModelUnavailable(
                "TensorFlow und TensorFlow Hub sind nicht installiert. "
                "Installiere zusätzlich requirements-ml.txt."
            ) from exc
        del tf, hub, audio_bytes, filename
        raise ModelUnavailable(
            "YAMNet ist vorgesehen, die Audio-Vorverarbeitung und das Label-Mapping "
            "werden in Schulstunde 3 ergänzt."
        )


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
    if model_key == "own":
        model_path = root / "models" / "model_a" / "keras_model.h5"
        labels_path = root / "models" / "model_a" / "labels.txt"
        if model_path.exists() and labels_path.exists():
            return {"ready": True, "message": "Exportierte Modell-Datei gefunden."}
        if model_path.exists() and not labels_path.exists():
            return {
                "ready": False,
                "message": "Modell gefunden, aber labels.txt fehlt noch.",
            }
        return {
            "ready": False,
            "message": "Noch kein Export vorhanden. Die Oberfläche nutzt den Demo-Modus.",
        }
    try:
        import tensorflow  # noqa: F401
        import tensorflow_hub  # noqa: F401
        return {"ready": True, "message": "TensorFlow und TensorFlow Hub sind verfügbar."}
    except ImportError:
        return {
            "ready": False,
            "message": "Optionale ML-Abhängigkeiten fehlen. Die Oberfläche nutzt den Demo-Modus.",
        }


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
