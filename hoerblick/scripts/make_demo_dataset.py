from __future__ import annotations

import argparse
import csv
import math
import random
import struct
import wave
from pathlib import Path


SAMPLE_RATE = 16_000
DURATION_SECONDS = 2.0
CLASS_KEYS = ("alarm", "siren", "doorbell", "background")


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _write_wave(path: Path, samples: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = b"".join(struct.pack("<h", int(_clip(value) * 32767)) for value in samples)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)


def _envelope(t: float, start: float, end: float, fade: float = 0.03) -> float:
    if t < start or t >= end:
        return 0.0
    return min(1.0, (t - start) / fade, (end - t) / fade)


def _alarm(t: float, rng: random.Random) -> float:
    if (t % 0.42) >= 0.28:
        return 0.0
    frequency = 720.0 + rng.uniform(-20.0, 20.0)
    return 0.34 * math.sin(2 * math.pi * frequency * t) + 0.14 * math.sin(
        2 * math.pi * frequency * 2 * t
    )


def _siren(t: float, rng: random.Random) -> float:
    phase = (t % 1.25) / 1.25
    sweep = 0.5 - 0.5 * math.cos(2 * math.pi * phase)
    frequency = 520.0 + 620.0 * sweep + rng.uniform(-12.0, 12.0)
    return 0.38 * math.sin(2 * math.pi * frequency * t)


def _doorbell(t: float, rng: random.Random) -> float:
    first = _envelope(t, 0.06, 0.62)
    second = _envelope(t, 0.72, 1.35)
    first_frequency = 660.0 + rng.uniform(-15.0, 15.0)
    second_frequency = 880.0 + rng.uniform(-15.0, 15.0)
    return 0.42 * first * math.sin(2 * math.pi * first_frequency * t) + (
        0.42 * second * math.sin(2 * math.pi * second_frequency * t)
    )


def _background(t: float, rng: random.Random) -> float:
    hum = 0.025 * math.sin(2 * math.pi * 95.0 * t)
    return hum + rng.uniform(-0.035, 0.035)


GENERATORS = {
    "alarm": _alarm,
    "siren": _siren,
    "doorbell": _doorbell,
    "background": _background,
}


def _make_samples(class_key: str, seed: int) -> list[float]:
    rng = random.Random(seed)
    generator = GENERATORS[class_key]
    amplitude = rng.uniform(0.82, 1.0)
    sample_count = int(SAMPLE_RATE * DURATION_SECONDS)
    return [
        amplitude * generator(index / SAMPLE_RATE, rng)
        for index in range(sample_count)
    ]


def create_dataset(
    project_root: Path,
    train_count: int,
    test_count: int,
    overwrite: bool,
) -> list[dict[str, str]]:
    metadata: list[dict[str, str]] = []
    for split, count in (("train", train_count), ("test", test_count)):
        for class_index, class_key in enumerate(CLASS_KEYS):
            for item_index in range(count):
                seed = 10_000 * (split == "test") + 1_000 * class_index + item_index
                filename = f"synthetic_{split}_{class_key}_{item_index + 1:02d}.wav"
                path = project_root / "data" / split / class_key / filename
                if overwrite or not path.exists():
                    _write_wave(path, _make_samples(class_key, seed))
                metadata.append(
                    {
                        "datei": str(path.relative_to(project_root)).replace("\\", "/"),
                        "split": split,
                        "klasse": class_key,
                        "quelle": "lokal erzeugter synthetischer Prototyp-Ton",
                        "lizenz": "eigene Erzeugung",
                        "hinweis": "Nur für den technischen Prototyp; später durch echte Aufnahmen ersetzen.",
                    }
                )
    return metadata


def write_metadata(project_root: Path, rows: list[dict[str, str]]) -> None:
    metadata_path = project_root / "data" / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Erzeugt einen klar gekennzeichneten synthetischen HörBlick-Prototypdatensatz."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.train_count < 1 or args.test_count < 1:
        parser.error("train-count und test-count müssen mindestens 1 sein.")

    root = args.root.resolve()
    rows = create_dataset(
        root,
        train_count=args.train_count,
        test_count=args.test_count,
        overwrite=args.overwrite,
    )
    write_metadata(root, rows)
    print(
        f"Erzeugt: {args.train_count * len(CLASS_KEYS)} Trainingsdateien und "
        f"{args.test_count * len(CLASS_KEYS)} Testdateien."
    )
    print("Hinweis: Die Dateien sind synthetische Prototyp-Töne und kein Ersatz für echte Aufnahmen.")


if __name__ == "__main__":
    main()
