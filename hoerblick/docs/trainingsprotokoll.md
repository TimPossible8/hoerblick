# Trainingsprotokoll: eigenes Modell

## Sitzung

- Schulstunde: 2
- Datum: 10. November 2026
- Werkzeug: lokales TensorFlow/Keras-Training mit Log-Mel-Spektrogrammen
- Modellname: HörBlick Modell A
- Exportziel: `models/model_a/`

## Klassen

| Klasse | Trainingsdateien | Testdateien | Aufnahmebedingungen |
|---|---:|---:|---|
| alarm | 7 | 3 | Quelldateien aus `audio hörblick/Feueralarm` |
| siren | 7 | 3 | Quelldateien aus `audio hörblick/sirenen` |
| doorbell | 11 | 4 | Quelldateien aus `audio hörblick/Klingeln` |
| background | 5 | 2 | Quelldateien aus `audio hörblick/Hintergrund` |

## Trainingsbedingungen

- Eingangsformat: WAV, MP3, FLAC und M4A (M4A wird lokal mit FFmpeg dekodiert)
- Zielklassen: `alarm`, `siren`, `doorbell`, `background`
- Ein reproduzierbarer Split mit Seed 2627 hält 12 Originaldateien dateiweise als Testdaten zurück.
- Räume, Lautstärken und Abstände sollen möglichst variiert werden.
- Exportformat: Keras HDF5 (`models/model_a/keras_model.h5`)
- Die Quelldateien wurden in `data/real_audio/` nach Trainings- und Testsatz kopiert.
- Trainingslauf: 8 Epochen bis Early Stopping, Test-Accuracy 25 % (3/12).
- Recall im Testsatz: Alarm 100 % (3/3), Sirene 0 % (0/3), Türklingel 0 % (0/4), Hintergrund 0 % (0/2).
- Das Modell sagte für alle zwölf Testdateien `alarm` voraus. Es ist daher nur ein technischer Trainings- und Integrationsprototyp und für Warnungen ungeeignet.
- Herkunft und Lizenz jeder Quelldatei müssen vor einer Weitergabe einzeln geprüft werden.

## Exportkontrolle

- [x] `models/model_a/keras_model.h5` vorhanden
- [x] `models/model_a/labels.txt` vorhanden
- [x] Labelreihenfolge mit den vier Projektklassen abgeglichen
- [x] getrennten Testdatensatz ausgewertet; Messwerte in `models/model_a/metrics.json`
- [x] Fehlklassifikationen dokumentiert
- [ ] ausreichend neue Hintergrund- und Sirenenaufnahmen sammeln und erneut trainieren

## Beobachtungen und Grenzen

Das exportierte Modell lässt sich lokal über den HörBlick-Modelladapter laden.
Der erste getrennte Test zeigt starke Fehlklassifikation: Das Modell gibt für
alle Testdateien die Klasse `alarm` aus. Der kleine Datensatz und die ungleiche
Klassenverteilung reichen nicht für eine brauchbare Erkennung. Das Modell darf
nicht als Sicherheitslösung oder verlässliche Warntechnik verwendet werden.

## Nächster Schritt

Weitere rechtlich geklärte Aufnahmen sammeln, besonders für `background` und
`siren`. Danach mit `python scripts/train_keras_model.py` neu trainieren und den
getrennten Testlauf in `models/model_a/metrics.json` kontrollieren.
