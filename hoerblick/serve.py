"""Serve the existing HörBlick design and model API on loopback only."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.parse import parse_qs, urlsplit

from hoerblick.models import ModelUnavailable, get_model_adapter, model_status
from hoerblick.priority import warning_for_prediction

ROOT = Path(__file__).resolve().parent
WEB = ROOT.parent / "web-prototype"
MODEL_LOCK = Lock()
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/script.js": ("script.js", "application/javascript; charset=utf-8"),
    "/streamlit.js": ("streamlit.js", "application/javascript; charset=utf-8"),
}


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, content_type="application/json; charset=utf-8"):
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in STATIC:
            filename, mime = STATIC[path]
            return self.reply(200, (WEB / filename).read_bytes(), mime)
        if path == "/api/status":
            with MODEL_LOCK:
                statuses = {key: model_status(key, ROOT) for key in ("own", "yamnet")}
            return self.reply(200, statuses)
        if path == "/api/comparison":
            with (ROOT / "vergleich.csv").open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            report = ROOT / "docs" / "vergleich_lauf.json"
            metadata = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
            return self.reply(200, {"rows": rows, "metadata": metadata})
        if path == "/api/comparison.csv":
            return self.reply(200, (ROOT / "vergleich.csv").read_bytes(), "text/csv; charset=utf-8")
        return self.reply(404, {"error": "Nicht gefunden."})

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path != "/api/analyse":
            return self.reply(404, {"error": "Nicht gefunden."})
        # Require a same-origin browser request. The API never enables CORS.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.reply(403, {"error": "Fremde Webseite nicht zugelassen."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.reply(400, {"error": "Ungültige Dateigröße."})
        if not 0 < length <= MAX_UPLOAD_BYTES:
            self.close_connection = True
            return self.reply(413, {"error": "Bitte eine Datei zwischen 1 Byte und 25 MB wählen."})
        model_key = parse_qs(url.query).get("model", [""])[0]
        if model_key not in ("own", "yamnet"):
            return self.reply(400, {"error": "Unbekannte Modellvariante."})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/octet-stream":
            return self.reply(415, {"error": "Audiodaten als application/octet-stream senden."})
        audio = self.rfile.read(length)
        try:
            with MODEL_LOCK:
                started = perf_counter()
                result = get_model_adapter(model_key, ROOT).predict(audio, "upload")
                elapsed = (perf_counter() - started) * 1000
            return self.reply(200, {
                **asdict(result), "warning": warning_for_prediction(result.label, result.confidence),
                "elapsed_ms": round(elapsed, 1), "clip_seconds": 5,
            })
        except ValueError as exc:
            return self.reply(400, {"error": str(exc)})
        except ModelUnavailable as exc:
            return self.reply(503, {"error": str(exc)})
        except Exception:
            logging.exception("Analyse fehlgeschlagen")
            return self.reply(500, {"error": "Analyse fehlgeschlagen. Details im Server-Terminal."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"HörBlick: http://127.0.0.1:{args.port} — Beenden mit Strg+C", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
