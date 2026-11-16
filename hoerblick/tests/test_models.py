import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from hoerblick.models import decode_audio, map_yamnet_scores, model_status
from hoerblick.priority import warning_for_prediction


class AudioAndMappingTests(unittest.TestCase):
    def test_stereo_resample_and_shared_duration(self):
        source = io.BytesIO()
        sf.write(source, np.column_stack([np.ones(44100 * 6) * .5, np.zeros(44100 * 6)]),
                 44100, format="WAV", subtype="PCM_16")
        audio = decode_audio(source.getvalue())
        self.assertEqual(audio.shape, (80000,))
        self.assertEqual(audio.dtype, np.float32)
        self.assertAlmostEqual(float(audio[100:-100].mean()), .25, places=3)

    def test_bad_audio_rejected(self):
        with self.assertRaises(ValueError):
            decode_audio(b"not audio")

    def test_empty_audio_rejected(self):
        source = io.BytesIO()
        sf.write(source, np.array([], dtype=np.float32), 16000, format="WAV")
        with self.assertRaises(ValueError):
            decode_audio(source.getvalue())

    def test_exact_alarm_name_and_mean_across_frames(self):
        result = map_yamnet_scores(
            [[.7, .2, .1], [.6, .1, .9]], ["Smoke detector, smoke alarm", "Doorbell", "Speech"])
        self.assertEqual(result.label, "alarm")
        self.assertAlmostEqual(result.confidence, .65)
        self.assertEqual(result.source_label, "Smoke detector, smoke alarm")

    def test_unmapped_top_class_is_background(self):
        result = map_yamnet_scores([[.8, .7]], ["Speech", "Siren"])
        self.assertEqual(result.label, "background")
        self.assertEqual(result.confidence, .8)

    def test_emergency_vehicle_and_ding_dong(self):
        for name, expected in [("Ambulance (siren)", "siren"), ("Ding-dong", "doorbell")]:
            self.assertEqual(map_yamnet_scores([[.9]], [name]).label, expected)

    def test_missing_model_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(model_status("own", Path(directory))["ready"])

    def test_runtime_features_match_training_pipeline(self):
        import tensorflow as tf
        from hoerblick.models import _prepare_audio_input
        from scripts.train_keras_model import build_log_mel, summarize_log_mel

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            wave = .2 * np.sin(2 * np.pi * 880 * np.arange(22050 * 2) / 22050)
            sf.write(path, np.column_stack([wave, wave * .5]), 22050)
            expected = summarize_log_mel(build_log_mel(path))
            actual = _prepare_audio_input(tf, decode_audio(path.read_bytes()), (None, 448)).numpy()[0]
            np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_uncertainty_does_not_give_green(self):
        for label in ("alarm", "siren", "doorbell", "background"):
            self.assertEqual(warning_for_prediction(label, .2)["level"], "Gelb")


if __name__ == "__main__":
    unittest.main()
