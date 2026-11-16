# HörBlick

HörBlick ist ein schulisches Prototyp-Projekt zur Klausurersatzleistung Informatik. Die Anwendung ordnet Umgebungsgeräusche ein und macht sicherheitsrelevante Schallereignisse für Menschen mit Hörbeeinträchtigung sichtbar.

Das Repository enthält eine gemeinsame Weboberfläche für den lokalen Python-Server und Streamlit, Trainings- und Testdaten, Modellcode sowie die Projektdokumentation. Die Anwendung ist eine Demonstration und ersetzt keine zertifizierte Warntechnik oder normgeprüften Rauchmelder.

## Projektstruktur

```text
.
├── web-prototype/       Statische HörBlick-Oberfläche
├── hoerblick/           Streamlit-App, ML-Module und Datensätze
│   ├── data/            Synthetische Trainings- und Testdaten
│   ├── docs/            Dokumentation und Reflexion
│   ├── models/          Zielordner für lokale Modellexporte
│   ├── scripts/         Datenprüfung und Training
│   └── tests/           Automatisierte Tests
└── docs/screenshots/    UI-Screenshots und PDF-Übersicht
```

## Webprototyp öffnen

`web-prototype/index.html` enthält die aktuelle mobile Oberfläche mit den drei Ansichten Analyse, Vergleich und Über. Für die lokale Audioanalyse `Start-HoerBlick.ps1` starten und `http://127.0.0.1:8765` öffnen. Ein Doppelklick auf die HTML-Datei zeigt nur die Oberfläche ohne Python-Anbindung. Die Schriftart und die Material-Symbole werden über Google Fonts geladen.

Die Bildschirmansichten sind in [docs/screenshots](docs/screenshots/) dokumentiert:

![Analyseansicht](docs/screenshots/01-analyse.png)

## Streamlit-App starten

Voraussetzung ist Python 3.11 oder neuer.

```powershell
cd hoerblick
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` bindet die Modellabhängigkeiten aus `requirements-ml.txt` ein, einschließlich TensorFlow und TensorFlow Hub.

Streamlit lädt dieselben HTML-, CSS- und JavaScript-Dateien aus `web-prototype/` als Komponente. Uploads werden über Streamlit an die vorhandenen Python-Modelladapter weitergereicht. Ein zusätzlicher HTTP-Server ist dafür nicht erforderlich.

Fehlende Modelle werden als nicht verfügbar angezeigt; es werden keine Demo-Ergebnisse als echte Analyse ausgegeben. Der eigene Export gehört nach `hoerblick/models/model_a/keras_model.h5` und erwartet 448 Merkmale und vier Klassen. Gewichte werden nicht in Git versioniert. YAMNet wird beim ersten Start aus TensorFlow Hub geladen. Ist nur YAMNet verfügbar, wird es automatisch ausgewählt.

## Auf Streamlit veröffentlichen

- Repository: `TimPossible8/hoerblick`
- Branch: `main`
- Main file path: `hoerblick/app.py`
- Python: `3.12`

Das gesamte Repository muss ausgecheckt werden, einschließlich `web-prototype/`. Nur `app.py` hochzuladen reicht nicht. Nach Aktualisierung der verbundenen Branch den Deployment-Stand prüfen und die App bei Bedarf neu starten.

In der Cloud werden Uploads auf dem Streamlit-Server verarbeitet. Die App schreibt sie nicht als Dateien auf die Festplatte. Ohne separat bereitgestellten eigenen Modellexport ist online nur YAMNet verfügbar. Die Vergleichsansicht zeigt die gespeicherten Messungen aus `hoerblick/vergleich.csv`, nicht einen neuen Cloud-Testlauf.

## Daten und Modi

Der eingecheckte Datensatz besteht aus 48 synthetischen Trainings- und 40 synthetischen Testdateien aufgeteilt auf die Klassen `alarm`, `siren`, `doorbell` und `background`. Die Töne dienen der Prüfung der Ordnerstruktur und des technischen Ablaufs. Für eine belastbare Bewertung sind echte, unterschiedlich aufgenommene und lizenzrechtlich geklärte Geräusche erforderlich.

Die Datensätze sind nach `train` und `test` getrennt. Herkunft und Lizenz jeder Datei stehen in [`hoerblick/data/metadata.csv`](hoerblick/data/metadata.csv). Details beschreibt [`hoerblick/data/README.md`](hoerblick/data/README.md).

## Prüfen und trainieren

Aus dem Ordner `hoerblick`:

```powershell
python scripts/check_dataset.py
python -m unittest discover -s tests -v
python scripts/make_demo_dataset.py
```

Das lokale Training erwartet die Quelldateien in `audio-source/` mit den Unterordnern `Feueralarm`, `sirenen`, `Klingeln` und `Hintergrund`. Der Pfad kann mit `--source` geändert werden. Das Training benötigt die Abhängigkeiten aus `requirements-ml.txt` und scikit-learn.

Reale Audioaufnahmen bleiben außerhalb von Git. Das Training übernimmt später die jeweils geklärten Dateien aus dem lokalen Quellordner und dokumentiert deren Herkunft in der Metadatendatei. Der kleine Modellcheck und seine Messwerte werden mit versioniert.

## Dokumentation

- [Projektdokumentation](hoerblick/docs/dokumentation.md)
- [Trainingsprotokoll](hoerblick/docs/trainingsprotokoll.md)
- [KI-Reflexion](hoerblick/docs/ki_reflexion.md)
- [UI-Screenshots](docs/screenshots/HoerBlick-UI-Screenshots.pdf)

## Technische Hinweise

Die Oberfläche zeigt eine klare Warnhierarchie mit Farbe, Text und Symbol. Ergebnisse beruhen auf dem gewählten Modell. Das Repository speichert keine erzeugten Modellgewichte, Zugangsdaten oder lokalen virtuellen Umgebungen.

Die verwendeten Webfonts und Material Symbols werden von Google Fonts geladen. YAMNet stammt aus TensorFlow Hub und steht unter Apache-2.0.
