# HörBlick

HörBlick ist ein Streamlit-Prototyp für die Klausurersatzleistung Informatik. Die Anwendung ordnet Umgebungsgeräusche ein und macht sicherheitsrelevante Schallereignisse für Menschen mit Hörbeeinträchtigung sichtbar.

Die Gestaltung orientiert sich an Material 3 Expressive: klare Typografie, kontrastreiche Warnstufen, funktionale Farbcodierung, große Bedienelemente und eine ruhige Oberfläche mit einzelnen expressiven Akzenten.

## Stand nach der dritten Schulstunde

Das fertige mobile Design ist mit echten Modellläufen verbunden:

- Web-Oberfläche mit Upload, Modellwahl, Warnstufe und Vergleichsdownload
- vorhandenes Keras-Modell (Extra Trees mit Keras-Kalibrierung)
- vortrainiertes YAMNet mit überprüftem AudioSet-Mapping
- gemeinsame Eingabe: erste fünf Sekunden, Mono, 16 kHz
- echte Fehleranzeigen statt automatisch erzeugter Demoergebnisse
- reproduzierbarer Vergleich mit `scripts/compare_models.py`

Die vollständige Umsetzung und Grenzen stehen in
[docs/schulstunde_3.md](docs/schulstunde_3.md).

### Fertiges Design starten

Im Projekt-Hauptordner `Start-HoerBlick.ps1` ausführen und
`http://127.0.0.1:8765` im Browser öffnen. Alternativ aus diesem Ordner:

```powershell
.\.venv-ml312\Scripts\python.exe serve.py
```

Diese geprüfte Laufzeit verwendet Python 3.12. Die alte `.venv` enthält
inkompatible 3.12-Binärpakete bei einem 3.13-Interpreter. `.venv-ml312` wird
deshalb für den lokalen Start bevorzugt. Das vorhandene Modell bleibt
in `models/model_a/`. YAMNet wird beim ersten Laden aus TensorFlow Hub
heruntergeladen; danach wird der lokale Cache genutzt.

Der HTML-Doppelklick allein startet keinen Python-Server.
Ohne Server wird „Server nicht erreichbar“ angezeigt.

### Installation auf einem anderen Computer

Python 3.12 verwenden, dann:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-ml.txt
.\.venv\Scripts\python.exe serve.py
```

`labels.txt` ist eingecheckt. Wenn `keras_model.h5` lokal fehlt, wird der
veröffentlichte Export automatisch von GitHub Releases geladen und anhand
von `models/model_a/release.json` auf Größe und SHA-256 geprüft. Die Datei
bleibt im ignorierten Ordner `models/model_a/.cache/`. Ein vorhandener lokaler
Trainingsexport hat Vorrang. Es ist kein erneutes Training notwendig.

### Streamlit-Alternative und Vergleich

```powershell
.\.venv-ml312\Scripts\python.exe -m streamlit run app.py
.\.venv-ml312\Scripts\python.exe scripts/compare_models.py
.\.venv-ml312\Scripts\python.exe -m unittest discover -s tests -v
```

### Online-Demo mit Streamlit Community Cloud

Beim Deployment `TimPossible8/hoerblick`, Branch `main`, Hauptdatei
`hoerblick/app.py` und Python `3.12` auswählen. `requirements.txt` installiert
auch die ML-Abhängigkeiten. Das eigene Modell wird beim ersten Start aus dem
GitHub-Release `model-a-2026-11-16` geladen, YAMNet aus TensorFlow Hub.
Beide Downloads werden zwischengespeichert. Es sind keine Zugangsdaten für
die Modelldownloads nötig. Fehler erscheinen ausdrücklich in der Oberfläche.

Streamlit und `serve.py` verwenden dieselbe mobile Oberfläche aus
`web-prototype/` und dieselben Modelladapter. Upload-Dateien werden nicht
gespeichert. In der Cloud erfolgt die Audioberechnung auf dem Streamlit-Server.

## Basis aus den ersten beiden Schulstunden

Die technische Basis ist vorbereitet:

- Streamlit-Grundgerüst mit den Seiten `Analyse`, `Vergleich` und `Über`
- vorbereitete Datenstruktur für Training und Test
- Trainingsdatenprüfung mit getrennten Klassen- und Testordnern
- Trainingsprotokoll für das eigene lokal trainierte Modell
- Keras-Import mit `keras_model.h5` und `labels.txt`
- Adapter-Struktur für das eigene Modell und YAMNet
- transparente Prioritäts- und Warnstufenlogik
- vorbereitete Vergleichstabelle
- README, Dokumentationsgerüst und KI-Reflexion

Der frühere Demo-Adapter bleibt im Code für Entwicklungszwecke vorhanden.
Die nutzbaren Oberflächen erzeugen bei Modellfehlern keine Demo-Vorhersage.

Für den Start gelten die oben beschriebenen Befehle mit Python 3.12 und
beiden Abhängigkeitsdateien. `requirements-tested.txt` hält die tatsächlich
geprüften Paketversionen fest.

## Datenstruktur

```text
data/
  train/
    alarm/
    siren/
    doorbell/
    background/
  test/
    alarm/
    siren/
    doorbell/
    background/
```

Der Testdatensatz darf nicht für das Training verwendet werden. Vor der Modellphase werden pro Klasse möglichst unterschiedliche Räume, Lautstärken und Abstände gesammelt.

## Schulstunde 2: eigenes Modell

Die Datenprüfung und der Exportpfad sind vorbereitet:

```powershell
python scripts/check_dataset.py
```

Das lokale Keras-Modell wird mit den Aufnahmen aus `audio-source/` trainiert.
Der Befehl erstellt
reproduzierbar einen dateiweisen Train/Test-Split und exportiert das Modell:

```powershell
python scripts/train_keras_model.py
```

Für einen lokal reproduzierbaren technischen Testdatensatz kann dieser Befehl
verwendet werden:

```powershell
python scripts/make_demo_dataset.py
```

Die erzeugten WAV-Dateien sind ausdrücklich synthetische Prototyp-Töne. Sie
erlauben die Prüfung von Ordnerstruktur und Upload-Ablauf, ersetzen aber keine
echten, unterschiedlich aufgenommenen Geräusche. Für eine belastbare
Bewertung müssen die Dateien später durch eigene oder lizenzrechtlich geklärte
Aufnahmen ersetzt werden.

Der Import liest die Labeldatei, lädt die Audiodatei mit Mono und 16 kHz und
bereitet die Audioeingabe passend zur Form des exportierten Keras-Modells vor.
Die Ergebnisse bleiben auch nach dem ersten Vergleich eine Demonstration:
Die kleine Testmenge belegt keine zuverlässige Erkennung im Alltag.

## Aktueller Modellcheck

Das lokale Modell in `models/model_a/` wurde mit 42 Aufnahmen trainiert: 30
Trainingsdateien und 12 dateiweise zurückgehaltene Testdateien. Die separate
Validierung innerhalb des Trainingssatzes erreicht 87,5 % (7/8); der HDF5-
Rundladetest bestätigt denselben Wert. Auf dem unangetasteten Testsatz erreicht
das exportierte Modell 66,7 % (8/12) und liegt damit unter 75 %. Die kleine
Validierungs- und Testmenge erlaubt keine belastbare Aussage zur Erkennung neuer
Aufnahmen; das Modell ist keine verlässliche Warntechnik.

Die großen Quelldateien und das abgeleitete Test-Array bleiben außerhalb von
Git. Die Messwerte stehen in `models/model_a/metrics.json`.

## Weiterer Ablauf

1. Weitere rechtlich geklärte Aufnahmen sammeln, besonders für `alarm` und `background`.
2. Den ersten gemeinsamen Vergleich auswerten und mit weiteren unabhängigen Aufnahmen erweitern.
3. Lizenzangaben der vorhandenen realen Aufnahmen vervollständigen.
4. Vergleichstabelle, Dokumentation und Pitch fertigstellen.

## Grenzen

Der Prototyp ist keine zertifizierte Sicherheitslösung und ersetzt keine normgeprüften Rauchmelder. Mikrofonreichweite, Hintergrundgeräusche, Fehlalarme, Akkulaufzeit und Datenschutz müssen bei der Bewertung berücksichtigt werden.
