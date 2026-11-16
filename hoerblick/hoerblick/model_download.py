"""Fetch the published model when no local training export is available."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen


def resolve_own_model(model_path: Path) -> Path:
    # Local training exports deliberately take precedence over the release.
    if model_path.is_file():
        return model_path

    manifest_path = model_path.with_name("release.json")
    if not manifest_path.is_file():
        raise ValueError("keras_model.h5 und die Download-Konfiguration release.json fehlen.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["sha256"]
    size = manifest["size_bytes"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not isinstance(size, int) or not 0 < size <= 50_000_000
            or not manifest["url"].startswith("https://github.com/TimPossible8/hoerblick/releases/download/")):
        raise ValueError("Ungültige Download-Konfiguration für das eigene Modell.")

    cache_dir = model_path.parent / ".cache"
    cached_path = cache_dir / f"{digest}.h5"
    if cached_path.is_file() and hashlib.sha256(cached_path.read_bytes()).hexdigest() == digest:
        return cached_path

    with urlopen(manifest["url"], timeout=60) as response:
        content = response.read(size + 1)
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Der Modelldownload stimmt nicht mit der veröffentlichten Prüfsumme überein.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".tmp", delete=False) as target:
            temporary_path = Path(target.name)
            target.write(content)
        temporary_path.replace(cached_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return cached_path
