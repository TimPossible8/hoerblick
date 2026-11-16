import base64
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from hoerblick.models import ModelResult, ModelUnavailable
from hoerblick.streamlit_bridge import handle_request

ROOT = Path(__file__).resolve().parents[1]


class StreamlitBridgeTests(unittest.TestCase):
    def test_upload_uses_real_adapter_and_returns_warning(self):
        adapter = Mock()
        adapter.predict.return_value = ModelResult("siren", .9, "YAMNet")
        with patch("hoerblick.streamlit_bridge.get_model_adapter", return_value=adapter):
            response = handle_request({
                "id": "upload-1", "action": "analyse", "model": "yamnet",
                "audio": base64.b64encode(b"audio").decode(),
            }, ROOT)
        self.assertEqual(response["id"], "upload-1")
        self.assertEqual(response["data"]["warning"]["level"], "Rot")
        self.assertFalse(response["data"]["is_demo"])
        adapter.predict.assert_called_once_with(b"audio", "upload")
        self.assertNotIn("audio", response)

    def test_rejects_invalid_uploads_before_model_call(self):
        cases = [
            {"model": "demo", "audio": "YQ=="},
            {"model": "own", "audio": ""},
            {"model": "own", "audio": "!!!"},
            {"model": "own", "audio": 123},
            {"model": "own", "audio": "YWFhYQ=="},  # Four bytes exceed patched limit.
        ]
        with patch("hoerblick.streamlit_bridge.MAX_UPLOAD_BYTES", 3), \
             patch("hoerblick.streamlit_bridge.get_model_adapter") as adapter:
            for payload in cases:
                with self.subTest(payload=payload):
                    self.assertIn("error", handle_request({"id": "bad", "action": "analyse", **payload}, ROOT))
            adapter.assert_not_called()

    def test_model_missing_is_reported_without_fake_result(self):
        adapter = Mock()
        adapter.predict.side_effect = ModelUnavailable("Modell fehlt")
        with patch("hoerblick.streamlit_bridge.get_model_adapter", return_value=adapter):
            response = handle_request({"id": "missing", "action": "analyse", "model": "own", "audio": "YQ=="}, ROOT)
        self.assertEqual(response, {"id": "missing", "error": "Modell fehlt"})

    def test_initial_data_has_comparison_csv_and_cloud_status(self):
        with patch("hoerblick.streamlit_bridge.model_status", return_value={"ready": True, "message": "lokal"}):
            response = handle_request({"id": "init", "action": "init"}, ROOT)
        data = response["data"]
        self.assertTrue(data["comparison"]["rows"])
        self.assertIn("Kriterium", data["csv"])
        self.assertIn("Server", data["statuses"]["yamnet"]["message"])

    def test_cloud_keeps_missing_own_model_unavailable(self):
        statuses = [
            {"ready": False, "message": "keras_model.h5 fehlt in models/model_a/."},
            {"ready": True, "message": "geladen"},
        ]
        with patch("hoerblick.streamlit_bridge.model_status", side_effect=statuses):
            response = handle_request({"id": "cloud", "action": "init"}, ROOT)
        self.assertFalse(response["data"]["statuses"]["own"]["ready"])
        self.assertTrue(response["data"]["statuses"]["yamnet"]["ready"])


if __name__ == "__main__":
    unittest.main()
