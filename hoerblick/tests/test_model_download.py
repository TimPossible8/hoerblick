import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from hoerblick.model_download import resolve_own_model
from hoerblick.models import ModelUnavailable, OwnModelAdapter


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "models/model_a/keras_model.h5"
        self.path.parent.mkdir(parents=True)
        self.content = b"test model export"
        self.digest = hashlib.sha256(self.content).hexdigest()
        self.path.with_name("release.json").write_text(json.dumps({
            "url": "https://github.com/TimPossible8/hoerblick/releases/download/test/keras_model.h5",
            "sha256": self.digest, "size_bytes": len(self.content),
        }), encoding="utf-8")
        self.path.with_name("labels.txt").write_text(
            "alarm\nsiren\ndoorbell\nbackground\n", encoding="utf-8")

    def test_download_once_and_reuse_verified_cache(self):
        with patch("hoerblick.model_download.urlopen", return_value=io.BytesIO(self.content)) as download:
            downloaded = resolve_own_model(self.path)
            self.assertEqual(downloaded.read_bytes(), self.content)
            self.assertEqual(resolve_own_model(self.path), downloaded)
            download.assert_called_once()
        self.assertFalse(self.path.exists())
        self.assertFalse(list(downloaded.parent.glob("*.tmp")))

    def test_local_training_export_takes_precedence(self):
        self.path.write_bytes(b"new local export")
        with patch("hoerblick.model_download.urlopen") as download:
            self.assertEqual(resolve_own_model(self.path), self.path)
            download.assert_not_called()

    def test_rejects_corrupt_truncated_and_oversized_downloads(self):
        for content in (b"x" * len(self.content), self.content[:-1], self.content + b"x"):
            with self.subTest(content=content):
                with patch("hoerblick.model_download.urlopen", return_value=io.BytesIO(content)):
                    with self.assertRaisesRegex(ValueError, "Prüfsumme"):
                        resolve_own_model(self.path)
                self.assertFalse(list(self.path.parent.rglob("*.h5")))

    def test_corrupt_cache_is_replaced(self):
        cached = self.path.parent / ".cache" / f"{self.digest}.h5"
        cached.parent.mkdir()
        cached.write_bytes(b"broken")
        with patch("hoerblick.model_download.urlopen", return_value=io.BytesIO(self.content)):
            self.assertEqual(resolve_own_model(self.path).read_bytes(), self.content)

    def test_network_failure_is_reported_and_can_be_retried(self):
        with patch("hoerblick.model_download.urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(ModelUnavailable, "heruntergeladen"):
                OwnModelAdapter(self.root).load()
        with patch("hoerblick.model_download.urlopen", return_value=io.BytesIO(self.content)):
            self.assertEqual(resolve_own_model(self.path).read_bytes(), self.content)

    def test_adapter_loads_downloaded_file(self):
        with patch("hoerblick.model_download.urlopen", return_value=io.BytesIO(self.content)), \
             patch("hoerblick.models._load_own_model", return_value="loaded") as load:
            model, labels = OwnModelAdapter(self.root).load()
        self.assertEqual(model, "loaded")
        self.assertEqual(labels, ["alarm", "siren", "doorbell", "background"])
        self.assertEqual(Path(load.call_args.args[0]).read_bytes(), self.content)


if __name__ == "__main__":
    unittest.main()
