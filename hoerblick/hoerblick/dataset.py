from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
CLASS_KEYS = ("alarm", "siren", "doorbell", "background")
SOURCE_EXTENSIONS = AUDIO_EXTENSIONS | {".m4a"}


@dataclass(frozen=True)
class ClassSummary:
    key: str
    train_count: int
    test_count: int
    train_files: tuple[str, ...]
    test_files: tuple[str, ...]

    @property
    def ready_for_training(self) -> bool:
        return self.train_count > 0

    @property
    def ready_for_testing(self) -> bool:
        return self.test_count > 0


def _audio_files(directory: Path) -> tuple[str, ...]:
    if not directory.exists():
        return ()
    return tuple(
        sorted(
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
    )


def inspect_dataset(project_root: Path) -> tuple[ClassSummary, ...]:
    data_root = project_root / "data"
    summaries: list[ClassSummary] = []
    for key in CLASS_KEYS:
        train_files = _audio_files(data_root / "train" / key)
        test_files = _audio_files(data_root / "test" / key)
        summaries.append(
            ClassSummary(
                key=key,
                train_count=len(train_files),
                test_count=len(test_files),
                train_files=train_files,
                test_files=test_files,
            )
        )
    return tuple(summaries)


def dataset_status(project_root: Path, minimum_per_class: int = 10) -> dict[str, object]:
    summaries = inspect_dataset(project_root)
    train_total = sum(summary.train_count for summary in summaries)
    test_total = sum(summary.test_count for summary in summaries)
    missing_train = [
        summary.key
        for summary in summaries
        if summary.train_count < minimum_per_class
    ]
    missing_test = [
        summary.key
        for summary in summaries
        if summary.test_count < minimum_per_class
    ]
    return {
        "summaries": summaries,
        "train_total": train_total,
        "test_total": test_total,
        "minimum_per_class": minimum_per_class,
        "ready_for_training": not missing_train,
        "ready_for_comparison": not missing_test,
        "missing_train": tuple(missing_train),
        "missing_test": tuple(missing_test),
    }
