import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from unittest.mock import Mock, patch

from serve import Handler, MAX_UPLOAD_BYTES
from hoerblick.models import ModelResult, ModelUnavailable


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        connection.request("POST" if body is not None else "GET", path, body=body, headers=headers or {})
        response = connection.getresponse()
        status, data = response.status, response.read()
        connection.close()
        return status, data

    def test_private_files_not_served(self):
        for path in ("/../hoerblick/models/model_a/keras_model.h5", "/api/unknown", "/serve.py"):
            self.assertEqual(self.request(path)[0], 404)

    def test_real_result_is_returned_with_warning(self):
        adapter = Mock()
        adapter.predict.return_value = ModelResult("alarm", .8, "Eigenes Modell")
        with patch("serve.get_model_adapter", return_value=adapter):
            status, body = self.request("/api/analyse?model=own", b"test",
                                        {"Content-Type": "application/octet-stream"})
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertFalse(result["is_demo"])
        self.assertEqual(result["warning"]["level"], "Rot")
        adapter.predict.assert_called_once_with(b"test", "upload")

    def test_model_error_does_not_return_demo(self):
        adapter = Mock()
        adapter.predict.side_effect = ModelUnavailable("Test: Modell fehlt")
        with patch("serve.get_model_adapter", return_value=adapter):
            status, body = self.request("/api/analyse?model=yamnet", b"test",
                                        {"Content-Type": "application/octet-stream"})
        self.assertEqual(status, 503)
        self.assertIn("error", json.loads(body))
        self.assertNotIn("confidence", json.loads(body))

    def test_cross_origin_and_oversize_rejected(self):
        self.assertEqual(self.request("/api/analyse?model=own", b"test",
                         {"Origin": "https://example.org"})[0], 403)
        self.assertEqual(self.request("/api/analyse?model=own", b"test",
                         {"Content-Length": str(MAX_UPLOAD_BYTES + 1)})[0], 413)


if __name__ == "__main__":
    unittest.main()
