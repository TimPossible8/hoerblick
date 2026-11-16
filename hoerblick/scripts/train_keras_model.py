from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hoerblick.dataset import AUDIO_EXTENSIONS, CLASS_KEYS
from hoerblick.models import ForestInference


SOURCE_ROOT = PROJECT_ROOT / "audio-source"
FOLDER_CLASSES = {
    "Feueralarm": "alarm",
    "sirenen": "siren",
    "Klingeln": "doorbell",
    "Hintergrund": "background",
}
SAMPLE_RATE = 16_000
CLIP_SECONDS = 5
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
N_MELS = 64
FRAME_LENGTH = 400
FRAME_STEP = 160
FFT_LENGTH = 512
SEED = 2627
TARGET_ACCURACY = 0.75
VALIDATION_SIZE = 0.25
INITIAL_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-4
LR_REDUCE_FACTOR = 0.5
LR_PATIENCE = 8
MIN_CALIBRATION_EPOCHS = 1


def build_log_mel(path: Path) -> np.ndarray:
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECONDS)
    except Exception:
        if path.suffix.lower() != ".m4a":
            raise
        import imageio_ffmpeg

        with tempfile.TemporaryDirectory() as directory:
            decoded = Path(directory) / "decoded.wav"
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(path),
                    "-t", str(CLIP_SECONDS), "-ac", "1", "-ar", str(SAMPLE_RATE),
                    str(decoded),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            audio, _ = librosa.load(decoded, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError(f"Leere Audiodatei: {path}")
    if audio.size < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - audio.size))
    else:
        audio = audio[:CLIP_SAMPLES]
    stft = tf.signal.stft(
        tf.convert_to_tensor(audio, dtype=tf.float32),
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
    )
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=FFT_LENGTH // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7_600.0,
    )
    log_mel = tf.math.log(tf.matmul(tf.abs(stft), mel_matrix) + 1e-6)
    return log_mel.numpy().astype(np.float32)


def collect_files(source_root: Path) -> dict[str, list[Path]]:
    by_class = {key: [] for key in CLASS_KEYS}
    for folder, class_key in FOLDER_CLASSES.items():
        class_dir = source_root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Klassenordner fehlt: {class_dir}")
        by_class[class_key] = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS | {".m4a"}
        )
        if len(by_class[class_key]) < 2:
            raise ValueError(f"Mindestens zwei Dateien für {class_key} erforderlich.")
    return by_class


def make_model(
    feature_count: int,
    class_count: int,
    normalization: tf.keras.layers.Normalization,
    classifier: ExtraTreesClassifier,
) -> tf.keras.Model:
    trees = []
    for estimator in classifier.estimators_:
        tree = estimator.tree_
        trees.append({
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "features": tree.feature.tolist(),
            "thresholds": tree.threshold.tolist(),
            "values": tree.value.reshape(tree.node_count, -1).tolist(),
            "max_depth": int(tree.max_depth),
        })

    inputs = tf.keras.Input(shape=(feature_count,), name="audio_summary_features")
    x = normalization(inputs)
    x = ForestInference(trees=trees, name="extra_trees_forest")(x)
    outputs = tf.keras.layers.Dense(
        class_count,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.Identity(),
        bias_initializer="zeros",
        name="trainable_class_calibration",
    )(x)
    model = tf.keras.Model(inputs, outputs, name="hoerblick_audio_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def summarize_log_mel(mel: np.ndarray) -> np.ndarray:
    delta = np.diff(mel, axis=0)
    return np.concatenate([
        mel.mean(axis=0),
        mel.std(axis=0),
        np.quantile(mel, [0.1, 0.5, 0.9], axis=0).ravel(),
        delta.mean(axis=0),
        delta.std(axis=0),
    ]).astype(np.float32)


def roundtrip_model(
    model: tf.keras.Model,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    learning_rate: float = INITIAL_LEARNING_RATE,
) -> tuple[tf.keras.Model, float]:
    custom_objects = {
        "ForestInference": ForestInference,
    }
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        temporary_model_path = Path(directory) / "keras_model.h5"
        model.save(temporary_model_path)
        loaded_model = tf.keras.models.load_model(
            temporary_model_path,
            compile=False,
            custom_objects=custom_objects,
        )
        loaded_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        predictions = loaded_model.predict(x_validation, verbose=0).argmax(axis=1)
        accuracy = float(np.mean(predictions == y_validation))
    return loaded_model, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainiert HörBlick Modell A lokal aus Audiodateien.")
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_ROOT,
        help="Ordner mit den Unterordnern Feueralarm, sirenen, Klingeln und Hintergrund.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.1 <= args.test_size <= 0.4:
        parser.error("--test-size muss zwischen 0.1 und 0.4 liegen.")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    files_by_class = collect_files(args.source)
    labels = list(CLASS_KEYS)
    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    records: list[dict[str, str]] = []
    metadata_path = PROJECT_ROOT / "data" / "dataset-metadata.csv"
    output_dir = PROJECT_ROOT / "models" / "model_a"
    held_out_path = output_dir / "held_out_test.npz"

    dataset_root = PROJECT_ROOT / "data" / "real_audio"
    for label_index, class_key in enumerate(labels):
        sources = files_by_class[class_key]
        train_paths, test_paths = train_test_split(
            sources,
            test_size=args.test_size,
            random_state=SEED,
        )
        split_by_path = {path: "train" for path in train_paths}
        split_by_path.update({path: "test" for path in test_paths})
        for path in sources:
            split = split_by_path[path]
            feature = summarize_log_mel(build_log_mel(path))
            all_features.append(feature)
            all_labels.append(label_index)
            records.append({
                "datei": str(
                    (dataset_root / split / class_key / path.name).relative_to(PROJECT_ROOT)
                ).replace("\\", "/"),
                "split": split,
                "klasse": class_key,
                "quelle": str(path),
                "lizenz": "Metadaten der Quelldatei prüfen; vor Abgabe dokumentieren",
            })

    features = np.stack(all_features)
    targets = np.asarray(all_labels, dtype=np.int32)
    train_indices = np.asarray([i for i, row in enumerate(records) if row["split"] == "train"])
    test_indices = np.asarray([i for i, row in enumerate(records) if row["split"] == "test"])
    x_train, y_train = features[train_indices], targets[train_indices]
    x_test, y_test = features[test_indices], targets[test_indices]
    fit_indices, validation_indices = train_test_split(
        np.arange(len(y_train)),
        test_size=VALIDATION_SIZE,
        random_state=SEED + 1,
        stratify=y_train,
    )
    x_fit, y_fit = x_train[fit_indices], y_train[fit_indices]
    x_validation, y_validation = x_train[validation_indices], y_train[validation_indices]

    normalization = tf.keras.layers.Normalization(name="feature_normalization")
    normalization.adapt(x_fit)
    normalized_fit = normalization(x_fit).numpy()
    classifier = ExtraTreesClassifier(
        n_estimators=500,
        max_features=1.0,
        class_weight="balanced",
        random_state=SEED,
    )
    classifier.fit(normalized_fit, y_fit)
    model = make_model(x_fit.shape[1], len(labels), normalization, classifier)
    best_validation_loss = float("inf")
    epochs_without_loss_improvement = 0
    epoch = 0
    rounds_completed = 0
    train_accuracy = float(
        np.mean(model.predict(x_fit, verbose=0).argmax(axis=1) == y_fit)
    )
    validation_loss, validation_accuracy = model.evaluate(
        x_validation, y_validation, verbose=0
    )
    print(
        f"Startbewertung (ExtraTrees): Genauigkeit Train {train_accuracy:.1%} | "
        f"Validierung {validation_accuracy:.1%} | "
        f"val_loss {validation_loss:.4f} | Lernrate "
        f"{INITIAL_LEARNING_RATE:.2g}",
        flush=True,
    )
    learning_rate = INITIAL_LEARNING_RATE

    while epoch < MIN_CALIBRATION_EPOCHS or validation_accuracy <= TARGET_ACCURACY:
        history = model.fit(
            x_fit,
            y_fit,
            validation_data=(x_validation, y_validation),
            epochs=epoch + 1,
            initial_epoch=epoch,
            batch_size=min(8, len(y_fit)),
            shuffle=True,
            verbose=0,
        )
        epoch += 1
        rounds_completed += 1
        train_accuracy = float(history.history["accuracy"][-1])
        validation_accuracy = float(history.history["val_accuracy"][-1])
        validation_loss = float(history.history["val_loss"][-1])
        learning_rate = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        print(
            f"Runde {rounds_completed} (Epoche {epoch}): "
            f"Genauigkeit Train {train_accuracy:.1%} | "
            f"Validierung {validation_accuracy:.1%} | "
            f"val_loss {validation_loss:.4f} | Lernrate {learning_rate:.2g}",
            flush=True,
        )

        if validation_accuracy > TARGET_ACCURACY:
            model, roundtrip_accuracy = roundtrip_model(
                model, x_validation, y_validation, learning_rate
            )
            if roundtrip_accuracy > TARGET_ACCURACY:
                validation_accuracy = roundtrip_accuracy
                print(
                    f"HDF5-Rundladetest bestanden: "
                    f"{validation_accuracy:.1%} Validierungsgenauigkeit.",
                    flush=True,
                )
                break
            print(
                f"HDF5-Rundladetest: {roundtrip_accuracy:.1%}; "
                "Training wird fortgesetzt.",
                flush=True,
            )
            validation_accuracy = roundtrip_accuracy

        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            epochs_without_loss_improvement = 0
        else:
            epochs_without_loss_improvement += 1

        if epochs_without_loss_improvement >= LR_PATIENCE:
            old_learning_rate = learning_rate
            new_learning_rate = max(
                old_learning_rate * LR_REDUCE_FACTOR,
                MIN_LEARNING_RATE,
            )
            if new_learning_rate < old_learning_rate:
                model.optimizer.learning_rate.assign(new_learning_rate)
                learning_rate = new_learning_rate
                print(
                    f"Adaptive Lernrate: {old_learning_rate:.2g} -> "
                    f"{new_learning_rate:.2g}",
                    flush=True,
                )
            else:
                new_learning_rate = INITIAL_LEARNING_RATE
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=new_learning_rate
                    ),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
                learning_rate = new_learning_rate
                print(
                    f"Adaptive Lernraten-Neustart: "
                    f"{old_learning_rate:.2g} -> {new_learning_rate:.2g}",
                    flush=True,
                )
            epochs_without_loss_improvement = 0

    print(
        f"Ziel erreicht nach {rounds_completed} Runden und {epoch} Keras-Epochen: "
        f"Validierungsgenauigkeit {validation_accuracy:.1%}.",
        flush=True,
    )
    scores = model.predict(x_test, verbose=0)
    predictions = scores.argmax(axis=1)
    confusion = tf.math.confusion_matrix(y_test, predictions, num_classes=len(labels)).numpy()
    accuracy = float(np.mean(predictions == y_test))
    per_class = {}
    for index, class_key in enumerate(labels):
        support = int(np.sum(y_test == index))
        true_positives = int(confusion[index, index])
        per_class[class_key] = {
            "test_count": support,
            "recall": round(true_positives / support, 4) if support else None,
        }

    model_path = output_dir / "keras_model.h5"
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary_directory:
        temporary_model_path = Path(temporary_directory) / "keras_model.h5"
        model.save(temporary_model_path)
        os.replace(temporary_model_path, model_path)

    for row in records:
        destination = PROJECT_ROOT / row["datei"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_path = Path(row["quelle"])
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)
    np.savez_compressed(held_out_path, features=x_test, labels=y_test)
    labels_path = output_dir / "labels.txt"
    labels_path.write_text(
        "".join(f"{index} {label}\n" for index, label in enumerate(labels)),
        encoding="utf-8",
    )
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = {
        "seed": SEED,
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "target_validation_accuracy": TARGET_ACCURACY,
        "validation_accuracy_at_save": round(validation_accuracy, 4),
        "classifier": "extra_trees_with_trainable_keras_calibration",
        "tree_count": len(classifier.estimators_),
        "epochs_completed": epoch,
        "training_rounds": rounds_completed,
        "fit_count": len(y_fit),
        "validation_count": len(y_validation),
        "initial_learning_rate": INITIAL_LEARNING_RATE,
        "final_learning_rate": round(learning_rate, 8),
        "accuracy": round(accuracy, 4),
        "test_accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "train_count": len(train_indices),
        "test_count": len(test_indices),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nModell gespeichert: {model_path}", flush=True)


if __name__ == "__main__":
    main()
