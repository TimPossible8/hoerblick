# Schulstunde 3 – Modellintegration und nutzbare Oberfläche

Bearbeitung: 16. November 2026.

## Umsetzung

Das bestehende Design in `web-prototype/` verwendet jetzt einen lokalen
Python-Server (`serve.py`). Analyse, Vergleich und Über sind bedienbar.
Upload und Modellwahl lösen reale Inferenz aus. Fehler werden angezeigt;
es gibt keinen automatischen Ersatz durch ein synthetisches Ergebnis.
Die ursprüngliche Streamlit-App nutzt dieselben Modelladapter.

Die Modellgewichte wurden nicht neu trainiert. Das vorhandene Modell ist
ein lokal trainierter Extra-Trees-Klassifikator mit einer Keras-Kalibrierung,
kein Teachable-Machine-Export. Seine benutzerdefinierte ForestInference-Schicht
bleibt erhalten.

## Vorverarbeitung und Modell A

Beide Modelle bekommen dieselben ersten fünf Sekunden einer Datei: Mono,
16 kHz, float32. Längere Dateien werden ausdrücklich abgeschnitten.
WAV, MP3, FLAC und OGG werden über SoundFile gelesen. Leere und beschädigte
Dateien werden abgewiesen.

Modell A erwartet 448 Merkmale und vier Ausgabewerte. Wie beim Training
werden kurze Aufnahmen auf fünf Sekunden mit Nullen aufgefüllt und aus
64 Log-Mel-Bändern Mittelwerte, Standardabweichungen, Quantile und zeitliche
Differenzen berechnet. Die Klassenfolge lautet alarm, siren, doorbell,
background. Die allgemeinen 22,05-kHz-/1-Sekunden-Vorgaben in AGENTS.md
beschreiben nicht diesen konkreten Export; die trainierte Schnittstelle
bleibt deshalb maßgeblich.

## Modell B: YAMNet

Bezugsquelle: `https://tfhub.dev/google/yamnet/1`.
YAMNet wird einmalig heruntergeladen und unter `models/model_b/tfhub_cache`
zwischengespeichert. Das Modell liefert Frame-Scores, Embeddings und
Spektrogramm. Für die Klassifikation werden nur die Scores verwendet.
Sehr kurze Eingaben werden auf mindestens 15.600 Samples aufgefüllt.

Die Scores jeder AudioSet-Klasse werden zeitlich gemittelt. Anschließend
wird die Klasse mit dem höchsten Mittelwert ausgewählt und zugeordnet:

| YAMNet-Originalklassen | HörBlick |
|---|---|
| Smoke detector, smoke alarm; Fire alarm | alarm |
| Siren; Civil defense siren; Emergency vehicle; Police car (siren); Ambulance (siren); Fire engine, fire truck (siren) | siren |
| Doorbell; Ding-dong | doorbell |
| alle anderen Originalklassen | background |

Diese einfache Top-1-Regel kann gleichzeitig auftretende Geräusche übersehen.
„background“ ist eine Sammelklasse, kein Beweis für Stille oder Sicherheit.
YAMNet-Scores sind unabhängig pro Klasse; sie werden nicht addiert oder zu
einer künstlichen Vierklassen-Softmax umgerechnet. Angezeigt wird der Score
der ausgewählten Originalklasse, ausdrücklich keine kalibrierte
Trefferwahrscheinlichkeit. Dieser ist nicht direkt mit Modell A vergleichbar.

## Warnstufen

Priorität = Gefahrenwert × Dringlichkeitswert × Modellscore.
Die vorhandenen Klassenfaktoren bleiben erhalten. Bei Score unter 0,35
erscheint Gelb mit „Unsicher“. Darüber erscheint Hintergrund grün, Alarm
oder Sirene ab Priorität 0,45 rot, sonst gelb. Grün bedeutet nur „kein
Zielsignal erkannt“, keine Entwarnung.

Die Schwellwerte sind Entwurfsentscheidungen, nicht anhand des Testdatensatzes
optimiert. Bei einer späteren Optimierung wird ein separater Validierungssatz
benötigt.

## Erster gemeinsamer Vergleich

Aufruf: `python scripts/compare_models.py`.
Standardmäßig verwendet das Skript `data/real_audio/test/`, prüft alle vier
Klassen und lehnt byte-identische Überschneidungen mit dem Trainingsordner ab.
Ähnliche Ausschnitte derselben Aufnahme kann dieser Hash-Check nicht erkennen.
Der Testdatensatz wurde bereits bei der früheren Modellauswertung verwendet;
dies ist keine neue unabhängige Validierung. Überlappungen mit dem
AudioSet-Training von YAMNet sind unbekannt.

Die Modellgewichte und Schwellwerte werden durch den Vergleich nicht verändert.
Beide Varianten bekommen dieselben Bytes, Dateihashes werden protokolliert.
Dateizugriff, Modellladen und ein Aufwärmlauf liegen außerhalb der Zeitmessung.
Gemessen wird Dekodierung + Resampling + Features + Inferenz. Das ist ein
einzelner Durchlauf pro Datei, kein stabiler Performance-Benchmark.

Ergebnisse:

Der Lauf am 16.11.2026 umfasst zwölf reale Dateien (drei Alarm, drei Sirenen,
vier Klingeln, zwei Hintergrundclips) und 24 Modellvorhersagen.

| Messwert | Eigenes Keras-Modell | YAMNet |
|---|---:|---:|
| Accuracy | 66,7 % (8/12) | 50,0 % (6/12) |
| Recall Alarm | 66,7 % (2/3) | 66,7 % (2/3) |
| Recall Sirene | 100,0 % (3/3) | 66,7 % (2/3) |
| Fehlalarmrate bei Hintergrund | 50,0 % (1/2) | 50,0 % (1/2) |
| Median Analysezeit | 2181,14 ms | 42,14 ms |

In dieser Stichprobe erkennt das eigene Modell mehr Klassen korrekt; YAMNet
ist deutlich schneller. Daraus folgt keine allgemeine Überlegenheit eines
Modells. Schon eine weitere richtig oder falsch erkannte Datei verändert
die Accuracy um etwa 8,3 Prozentpunkte. Die Fehlalarmrate beruht auf nur
zwei Hintergrundclips. Das eigene Modell erreicht hier nicht das angestrebte
Zeitbudget von 100 ms.

Gespeicherte Nachweise:

- `docs/testprotokoll.csv`: je Datei erwartete/vorhergesagte Klasse, Score,
  Modell, Warnstufe, Originalklasse, Hash und Laufzeit.
- `vergleich.csv`: Accuracy, Recall für Alarm/Sirene, Fehlalarmrate und Medianzeit.
- `docs/vergleich_lauf.json`: Datum, Datensatzgröße, Klassenverteilung,
  Modellhash und Definitionen.

Fehlalarmrate = Anteil gelber/roter Anzeigen unter echten Hintergrundclips,
einschließlich unsicherer Ergebnisse. Die wenigen Aufnahmen erlauben keine
allgemeine Leistungsbehauptung. Lizenzangaben der bereits vorhandenen
Realaufnahmen sind in `data/dataset-metadata.csv` weiterhin als zu klären
markiert; sie werden durch diese Integration nicht als freigegeben behauptet.

## Design, Datenschutz und Betrieb

Das mobile Design behält Farben, große Rundungen, segmentierte Modellwahl
und untere Dreiernavigation bei. Ergebnis, Farbe, Symbol und Text entsprechen
dem realen Lauf. Ein Wechsel von Datei/Modell oder Zurücksetzen entfernt das
alte Ergebnis; auch verspätete Antworten überschreiben keine neue Auswahl.
Gelbe Warnungen wurden für besser lesbaren weißen Text dunkler gestaltet.

Der Webserver bindet ausschließlich an 127.0.0.1, bietet nur ausgewählte
Dateien/API-Endpunkte an und speichert Uploads nicht. Audio verlässt bei
lokalem Betrieb nicht diesen Computer. Beim ersten YAMNet-Laden ist Internet
erforderlich; Google Fonts im Design erzeugt ebenfalls externe Anfragen.
Das ist ein lokaler Schulprototyp, kein öffentlich bereitzustellender Server.

## Durchgeführte Prüfungen

- 16 Unit-Tests bestanden: Audio-Vorverarbeitung, Übereinstimmung mit dem
  Trainingsskript, YAMNet-Mapping, Warnstufen und HTTP-Fehlerfälle.
- Beide Modelle geladen, aufgewärmt und auf denselben zwölf realen Dateien
  ausgeführt; 24 Vorhersagen im Testprotokoll.
- Reale HTTP-Anfragen an den laufenden Server mit beiden Modellvarianten
  erfolgreich; beschädigte Audiodaten liefern HTTP 400; Vergleichs-CSV abrufbar.
- Streamlit-AppTest: Analyse, Vergleich, Über und beide Modelloptionen ohne
  Exceptions.
- `pip check`: keine beschädigten Paketabhängigkeiten in `.venv-ml312`.
- Bestehende Datensatzprüfung: 48 synthetische Trainingsdateien und 40
  synthetische Testdateien. Diese wurden nicht als Leistungsbenchmark verwendet.
- Browserprüfung: drei Ansichten, Modellwechsel, reale Vergleichsdaten,
  Ansichten bei 320, 390 und 1280 Pixeln; keine protokollierten JavaScript-Fehler.
  Screenshots liegen unter `../../docs/screenshots/schulstunde3-*.jpg`.
- Einschränkung der UI-Prüfung: Der automatisierte Dateiauswahldialog im
  eingebauten Browser ließ sich nicht bedienen. Ein vollständiger Durchlauf
  „Datei auswählen → Ergebnis“ wurde dort deshalb nicht bestätigt. Die
  tatsächliche Audioverarbeitung über HTTP und beide Modelladapter wurde geprüft.

## Quellen und Lizenzen

Am 16.11.2026 gegen offizielle Projektquellen geprüft:

- YAMNet README / Architektur und Vorverarbeitung:
  `https://github.com/tensorflow/models/tree/master/research/audioset/yamnet`
- Originalklassen:
  `https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv`
- TensorFlow-Hub-Beispiel:
  `https://www.tensorflow.org/hub/tutorials/yamnet`
- TensorFlow Models: Apache License 2.0,
  `https://github.com/tensorflow/models/blob/master/LICENSE`
- AudioSet-Hintergrund:
  `https://research.google.com/audioset/`

Die Modell-/Code-Lizenz ersetzt nicht die Prüfung der Lizenzen einzelner
Testaufnahmen.
