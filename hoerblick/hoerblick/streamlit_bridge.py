"""Small request bridge for the shared web UI, without a second HTTP server."""
from __future__ import annotations

import base64
import binascii
import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from time import perf_counter

from .models import ModelUnavailable, get_model_adapter, model_status
from .priority import warning_for_prediction

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MODEL_LOCK = Lock()


def handle_request(request: dict, root: Path) -> dict:
    reply = {"id": request.get("id")}
    try:
        if request.get("action") == "init":
            with MODEL_LOCK:
                statuses = {key: model_status(key, root) for key in ("own", "yamnet")}
            for status in statuses.values():
                if status["ready"]:
                    status["message"] = "Modell geladen · Analyse auf dem Server"
            csv_text = (root / "vergleich.csv").read_text(encoding="utf-8-sig")
            report = root / "docs" / "vergleich_lauf.json"
            reply["data"] = {
                "statuses": statuses,
                "comparison": {
                    "rows": list(csv.DictReader(csv_text.splitlines())),
                    "metadata": json.loads(report.read_text(encoding="utf-8")) if report.exists() else {},
                },
                "csv": csv_text,
            }
        elif request.get("action") == "analyse":
            model = request.get("model")
            if model not in ("own", "yamnet"):
                raise ValueError("Unbekannte Modellvariante.")
            encoded = request.get("audio")
            if not isinstance(encoded, str) or not 0 < len(encoded) <= 4 * ((MAX_UPLOAD_BYTES + 2) // 3):
                raise ValueError("Bitte eine Datei zwischen 1 Byte und 25 MB wählen.")
            try:
                audio = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Ungültige Upload-Daten.") from exc
            if not 0 < len(audio) <= MAX_UPLOAD_BYTES:
                raise ValueError("Bitte eine Datei zwischen 1 Byte und 25 MB wählen.")
            with MODEL_LOCK:
                started = perf_counter()
                result = get_model_adapter(model, root).predict(audio, "upload")
                elapsed = (perf_counter() - started) * 1000
            reply["data"] = {
                **asdict(result),
                "warning": warning_for_prediction(result.label, result.confidence),
                "elapsed_ms": round(elapsed, 1),
                "clip_seconds": 5,
            }
        else:
            raise ValueError("Unbekannte Anfrage.")
    except (ValueError, ModelUnavailable) as exc:
        reply["error"] = str(exc)
    except Exception:
        logging.exception("Streamlit-Anfrage fehlgeschlagen")
        reply["error"] = "Anfrage fehlgeschlagen. Details stehen im Streamlit-Protokoll."
    return reply
