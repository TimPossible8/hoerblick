from __future__ import annotations

from pathlib import Path

import pandas as pd


COMPARISON_COLUMNS = [
    "Kriterium",
    "Eigenes Modell",
    "YAMNet",
]


def load_comparison_table(path: Path) -> pd.DataFrame:
    if path.exists():
        table = pd.read_csv(path)
        if list(table.columns) == COMPARISON_COLUMNS:
            return table
    return pd.DataFrame(
        [
            ["Accuracy", "noch nicht gemessen", "noch nicht gemessen"],
            ["Recall Alarm", "noch nicht gemessen", "noch nicht gemessen"],
            ["Recall Sirene", "noch nicht gemessen", "noch nicht gemessen"],
            ["Fehlalarmrate", "noch nicht gemessen", "noch nicht gemessen"],
            ["Analysezeit", "noch nicht gemessen", "noch nicht gemessen"],
            ["Trainings- und Integrationsaufwand", "zu dokumentieren", "zu dokumentieren"],
        ],
        columns=COMPARISON_COLUMNS,
    )
