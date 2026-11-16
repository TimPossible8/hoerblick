# Trainingsprotokoll: eigenes Modell

## Sitzung

- Schulstunde: 2
- Datum: 10. November 2026
- Werkzeug: lokales ExtraTrees-Modell mit trainierbarem Keras-Kalibrierungskopf auf Log-Mel-Merkmalen
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
- Von den 30 Trainingsdateien wurden 22 zum Fitten und 8 zur Validierung verwendet; die 12 Testdateien blieben vom Fitten und der Zielsteuerung ausgeschlossen.
- Nach einer Keras-Kalibrierungsepoche: Validierungsgenauigkeit 87,5 % (7/8); der HDF5-Rundladetest bestätigt 87,5 %. Das Trainingsziel von mehr als 75 % wurde damit auf der Validierungsmenge erreicht.
- Abschließender Test auf den zurückgehaltenen Dateien: 66,7 % Accuracy (8/12), unter 75 %.
- Recall im Testsatz: Alarm 66,7 % (2/3), Sirene 100 % (3/3), Türklingel 50 % (2/4), Hintergrund 50 % (1/2).
- Die adaptive Lernrate blieb in diesem Lauf bei 0,001, da das Trainingsziel nach der ersten Epoche erreicht war.
- Validierungs- und Testsatz umfassen nur wenige Dateien. Die Werte schwanken stark und belegen keine allgemeine Erkennungsleistung.
- Herkunft und Lizenz jeder Quelldatei müssen vor einer Weitergabe einzeln geprüft werden.

## Erweiterte externe Prüfung (11. November 2026)

- Die bestehende H5-Datei wurde unverändert auf vier neuen, aus dem FSD50K-
  Evaluierungssplit stammenden Originalaufnahmen geprüft. Die Original-IDs,
  Uploader, Titel und Clip-Lizenzen stehen in `data/fsd50k/metadata.csv`.
- Kandidaten wurden gegen FSD50K-Labels und Clip-Metadaten geprüft: Alarm nur,
  wenn FSD50K `Alarm` annotiert und Titel/Metadaten ausdrücklich Feueralarm
  benennen; Sirene nur bei `Siren`; Türklingel nur bei `Doorbell`. Wasserfall
  wurde als Hintergrund aufgenommen. Die vier Eval-Aufnahmen stammen von
  verschiedenen Uploadern und sind vom Training ausgeschlossen.
- Ergebnis der H5-Inferenz: **75,0 % (3/4)**. Feueralarm-ID 210513 wurde als
  Sirene klassifiziert; Hintergrund-ID 404010, Türklingel-ID 330400 und
  Sirene-ID 117120 wurden korrekt klassifiziert. Einzelwerte und Softmax-Werte
  stehen in `models/model_a/fsd50k_eval.json`.
- Die Messung liegt genau bei 75 %, also nicht über dem Ziel. Vier Clips sind
  keine robuste Genauigkeitsschätzung; außerdem zeigt der FSD50K-HF-Mirror nur
  einen sehr kleinen Teil der offiziellen Audiodateien. Das komplette offizielle
  Audioarchiv ist mehrere Gigabyte groß. Deshalb wurde die H5-Datei nicht mit
  diesen vier Clips abgestimmt und nicht als >75 % ausgewiesen.
- Zwei zusätzliche Trainingsclips (eine CC0-Sirene und eine CC0-Türklingel)
  wurden mit Quellenmetadaten geprüft. Sie wurden nicht in das Modell integriert,
  da zwei Clips ohne zusätzliche Alarme und Hintergrundaufnahmen keine
  sinnvolle, ausgeglichene Trainingserweiterung ergeben.
- Der aktuell verfügbare und überprüfte Quellenbestand reicht nicht aus, um
  >75 % auf einem angemessenen, unabhängigen Vier-Klassen-Test nachzuweisen.
  Benötigt werden weitere herunterladbare Originalclips je Klasse, insbesondere
  klar bezeichnete Rauch-/Feueralarme und vielfältige Hintergrundaufnahmen.

## Exportkontrolle

- [x] `models/model_a/keras_model.h5` vorhanden
- [x] `models/model_a/labels.txt` vorhanden
- [x] Labelreihenfolge mit den vier Projektklassen abgeglichen
- [x] getrennten Testdatensatz ausgewertet; Messwerte in `models/model_a/metrics.json`
- [x] Fehlklassifikationen dokumentiert
- [ ] ausreichend neue Hintergrund- und Sirenenaufnahmen sammeln und erneut trainieren

## Beobachtungen und Grenzen

Das exportierte Keras-Modell erreicht nach einer echten Kalibrierungsepoche
87,5 % auf der separaten Validierungsmenge und besteht den HDF5-Rundladetest.
Der unabhängige Abschlusstest erreicht jedoch nur 66,7 %. Die Validierungs-
schwelle von mehr als 75 % ist damit erreicht, die gleiche Leistung auf
zurückgehaltenen Audiodateien aber nicht nachgewiesen. Der kleine Datensatz und
die Klassenunterschiede reichen nicht für eine verlässliche Erkennung. Das
Modell darf nicht als Sicherheitslösung oder Warntechnik verwendet werden.

## Nächster Schritt

Weitere rechtlich geklärte Aufnahmen sammeln, besonders für `background` und
`siren`. Danach mit `python scripts/train_keras_model.py` neu trainieren und den
getrennten Testlauf in `models/model_a/metrics.json` kontrollieren.
