# HörBlick

HörBlick ist ein Streamlit-Prototyp für die Klausurersatzleistung Informatik. Die Anwendung ordnet Umgebungsgeräusche ein und macht sicherheitsrelevante Schallereignisse für Menschen mit Hörbeeinträchtigung sichtbar.

Die Gestaltung orientiert sich an Material 3 Expressive: klare Typografie, kontrastreiche Warnstufen, funktionale Farbcodierung, große Bedienelemente und eine ruhige Oberfläche mit einzelnen expressiven Akzenten.

## Stand nach der zweiten Schulstunde

Die technische Basis ist vorbereitet:

- Streamlit-Grundgerüst mit den Seiten `Analyse`, `Vergleich` und `Über`
- vorbereitete Datenstruktur für Training und Test
- Trainingsdatenprüfung mit getrennten Klassen- und Testordnern
- Trainingsprotokoll für das eigene Teachable-Machine-Modell
- Teachable-Machine-Keras-Import mit `keras_model.h5` und `labels.txt`
- Adapter-Struktur für das eigene Modell und YAMNet
- transparente Prioritäts- und Warnstufenlogik
- vorbereitete Vergleichstabelle
- README, Dokumentationsgerüst und KI-Reflexion

Solange keine echten Modelle installiert oder exportiert sind, läuft die App im eindeutig gekennzeichneten Demo-Modus. Dateinamen wie `alarm_demo.wav`, `siren_demo.wav` oder `doorbell_demo.wav` werden dafür automatisch erkannt. Der Trainingsstatus ist in der App unter `Training` sichtbar.

## Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Für die späteren Modellvarianten:

```powershell
python -m pip install -r requirements-ml.txt
```

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

Danach wird in Teachable Machine ein Audioprojekt mit den vier Klassen angelegt.
Nur die Dateien aus `data/train/` werden hochgeladen. Nach dem Training wird das
Keras-Modell nach `models/model_a/` exportiert. Dort müssen mindestens diese
beiden Dateien liegen:

```text
models/model_a/
  keras_model.h5
  labels.txt
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
Die Ergebnisse werden weiterhin als Demonstration behandelt, bis der Vergleich
mit dem getrennten Testdatensatz durchgeführt wurde.

## Aktueller Modellcheck

Der kleine lokale Modellcheck in `models/model_a/` ist eingecheckt. Er wurde mit
30 Trainings- und 12 Testdateien aus einem separaten realen Audioset erzeugt.
Die Accuracy liegt bei 25 %. Alle Testdateien wurden als `alarm` erkannt; die
Recall-Werte für Sirene, Klingel und Hintergrund sind deshalb 0 %. Das Ergebnis
ist ein dokumentierter Zwischenstand und keine belastbare Modellleistung.

Die großen Quelldateien und das abgeleitete Test-Array bleiben außerhalb von
Git. Die Messwerte stehen in `models/model_a/metrics.json`.

## Geplanter weiterer Ablauf

1. Eigenes Modell mit Teachable Machine trainieren und nach `models/model_a/` exportieren.
2. YAMNet mit TensorFlow Hub einbinden und auf die Projektklassen abbilden.
3. Beide Modellvarianten mit demselben Testdatensatz messen.
4. Vergleichstabelle, Dokumentation und Pitch fertigstellen.

## Grenzen

Der Prototyp ist keine zertifizierte Sicherheitslösung und ersetzt keine normgeprüften Rauchmelder. Mikrofonreichweite, Hintergrundgeräusche, Fehlalarme, Akkulaufzeit und Datenschutz müssen bei der Bewertung berücksichtigt werden.
