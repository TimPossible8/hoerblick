"""First reproducible, file-level comparison; no training or threshold fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hoerblick.models import get_model_adapter
from hoerblick.priority import CLASS_SPECS, warning_for_prediction


def summarize(rows):
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    background = [r for r in rows if r["expected"] == "background"]
    return {
        "accuracy": ratio(sum(r["expected"] == r["predicted"] for r in rows), len(rows)),
        "recall_alarm": ratio(sum(r["expected"] == r["predicted"] == "alarm" for r in rows),
                              sum(r["expected"] == "alarm" for r in rows)),
        "recall_siren": ratio(sum(r["expected"] == r["predicted"] == "siren" for r in rows),
                              sum(r["expected"] == "siren" for r in rows)),
        "false_alarm_rate": ratio(sum(r["warning"] != "Grün" for r in background), len(background)),
        "median_ms": statistics.median(r["elapsed_ms"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data/real_audio/test")
    parser.add_argument("--label", default="zurückgehaltene reale Aufnahmen")
    args = parser.parse_args()
    files = [(label, p) for label in CLASS_SPECS
             for p in sorted((args.test_dir / label).glob("*"))
             if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]
    if not files or any(not any(label == key for label, _ in files) for key in CLASS_SPECS):
        raise SystemExit("Ein Testordner mit Dateien in allen vier Klassen ist erforderlich.")
    # Refuse byte-identical train/test leakage where the sibling training set exists.
    train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (args.test_dir.parent / "train").rglob("*") if p.is_file()}
    for _, path in files:
        if hashlib.sha256(path.read_bytes()).hexdigest() in train_hashes:
            raise SystemExit(f"Train/Test-Duplikat: {path.name}")
    adapters = {key: get_model_adapter(key, ROOT) for key in ("own", "yamnet")}
    for key, adapter in adapters.items():
        adapter.load()
        adapter.predict(files[0][1].read_bytes(), "warmup.wav")
        print(f"{key}: geladen und aufgewärmt", flush=True)
    rows = []
    for expected, path in files:
        audio = path.read_bytes()
        for key, adapter in adapters.items():
            started = perf_counter()
            result = adapter.predict(audio, path.name)
            elapsed = (perf_counter() - started) * 1000
            if result.is_demo:
                raise RuntimeError("Demoergebnisse dürfen nicht bewertet werden.")
            rows.append({
                "file": path.relative_to(args.test_dir).as_posix(),
                "sha256": hashlib.sha256(audio).hexdigest(),
                "expected": expected, "predicted": result.label, "model": key,
                "confidence": round(result.confidence, 6),
                "warning": warning_for_prediction(result.label, result.confidence)["level"],
                "source_label": result.source_label, "elapsed_ms": round(elapsed, 2),
            })
        print(f"{expected}: {path.name}", flush=True)
    summaries = {key: summarize([r for r in rows if r["model"] == key]) for key in adapters}
    with (ROOT / "docs/testprotokoll.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    criteria = [
        ("Accuracy", "accuracy"), ("Recall Alarm", "recall_alarm"),
        ("Recall Sirene", "recall_siren"), ("Fehlalarmrate", "false_alarm_rate"),
        ("Analysezeit (Median)", "median_ms"),
    ]
    with (ROOT / "vergleich.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["Kriterium", "Eigenes Modell", "YAMNet"])
        for title, metric in criteria:
            values = [summaries[key][metric] for key in adapters]
            writer.writerow([title] + [
                "n/a" if v is None else (f"{v:.0f} ms" if metric == "median_ms" else f"{v:.1%}")
                for v in values
            ])
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_label": args.label, "test_dir": args.test_dir.resolve().as_posix(),
        "count": len(files), "class_counts": {k: sum(c == k for c, _ in files) for k in CLASS_SPECS},
        "window_seconds": 5, "sample_rate": 16000,
        "timing": "decode + resample + features + inference; model loading and warmup excluded",
        "false_alarm_definition": "yellow or red on a true background clip (including uncertain results)",
        "own_model_sha256": hashlib.sha256((ROOT / "models/model_a/keras_model.h5").read_bytes()).hexdigest(),
        "summaries": summaries,
        "limitations": "Small previously used held-out set; source licenses pending. Not a new independent benchmark. AudioSet training overlap unknown.",
    }
    (ROOT / "docs/vergleich_lauf.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
