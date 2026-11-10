from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassSpec:
    key: str
    label: str
    hazard: float
    urgency: float
    warning_level: str


CLASS_SPECS = {
    "alarm": ClassSpec("alarm", "Feueralarm", 1.00, 1.00, "Rot"),
    "siren": ClassSpec("siren", "Sirene", 0.85, 0.90, "Rot"),
    "doorbell": ClassSpec("doorbell", "Türklingel", 0.35, 0.55, "Gelb"),
    "background": ClassSpec("background", "Hintergrundgeräusch", 0.00, 0.00, "Grün"),
}

WARNING_LEVELS = {
    "Rot": {"icon": "!", "action": "Bitte sofort reagieren."},
    "Gelb": {"icon": "i", "action": "Ein Ereignis mit möglichem Handlungsbedarf."},
    "Grün": {"icon": "✓", "action": "Kein unmittelbarer Handlungsbedarf."},
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    aliases = {
        "fire alarm": "alarm",
        "smoke detector alarm": "alarm",
        "rauchmelder": "alarm",
        "feueralarm": "alarm",
        "emergency vehicle": "siren",
        "siren": "siren",
        "sirene": "siren",
        "doorbell": "doorbell",
        "türklingel": "doorbell",
        "background": "background",
        "hintergrund": "background",
        "silence": "background",
        "stille": "background",
    }
    return aliases.get(normalized, normalized if normalized in CLASS_SPECS else "background")


def priority_score(label: str, confidence: float) -> float:
    spec = CLASS_SPECS[normalize_label(label)]
    return round(spec.hazard * spec.urgency * max(0.0, min(1.0, confidence)), 4)


def warning_for_prediction(label: str, confidence: float) -> dict[str, str | float]:
    key = normalize_label(label)
    spec = CLASS_SPECS[key]
    score = priority_score(key, confidence)
    if key == "background" or confidence < 0.35:
        level = "Grün"
    elif score >= 0.45:
        level = "Rot" if key in {"alarm", "siren"} else "Gelb"
    else:
        level = "Gelb"
    details = WARNING_LEVELS[level]
    return {
        "level": level,
        "icon": details["icon"],
        "title": spec.label,
        "action": details["action"],
        "score": score,
    }
