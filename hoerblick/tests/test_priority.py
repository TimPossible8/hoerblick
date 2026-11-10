import unittest

from hoerblick.priority import normalize_label, priority_score, warning_for_prediction


class PriorityTests(unittest.TestCase):
    def test_aliases_are_normalized(self):
        self.assertEqual(normalize_label("Feueralarm"), "alarm")
        self.assertEqual(normalize_label("Doorbell"), "doorbell")
        self.assertEqual(normalize_label("unbekannt"), "background")


    def test_alarm_has_high_priority(self):
        self.assertGreater(priority_score("alarm", 0.9), priority_score("doorbell", 0.9))


    def test_warning_levels_are_accessible(self):
        self.assertEqual(warning_for_prediction("alarm", 0.9)["level"], "Rot")
        self.assertEqual(warning_for_prediction("doorbell", 0.9)["level"], "Gelb")
        self.assertEqual(warning_for_prediction("background", 0.9)["level"], "Grün")
