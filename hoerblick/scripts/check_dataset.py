from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hoerblick.dataset import dataset_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Prüft HörBlick-Trainings- und Testdaten.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="HörBlick-Projektordner",
    )
    parser.add_argument("--minimum", type=int, default=10)
    args = parser.parse_args()

    status = dataset_status(args.root, minimum_per_class=args.minimum)
    print(f"Trainingsdateien: {status['train_total']}")
    print(f"Testdateien: {status['test_total']}")
    print(f"Training bereit: {'ja' if status['ready_for_training'] else 'nein'}")
    print(f"Vergleich bereit: {'ja' if status['ready_for_comparison'] else 'nein'}")
    for summary in status["summaries"]:
        print(
            f"- {summary.key}: "
            f"train={summary.train_count}, test={summary.test_count}"
        )


if __name__ == "__main__":
    main()
