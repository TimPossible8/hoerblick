# Datenablage

Die Trainings- und Testdateien werden getrennt gehalten:

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

Der Testdatensatz darf in keiner Trainings- oder Validierungsphase verwendet werden. Für jede Datei werden Herkunft, Lizenz, Klasse und Aufnahmebedingungen dokumentiert.

## Prüfung

Mit folgendem Befehl werden die Klassen und Dateimengen kontrolliert:

```powershell
python scripts/check_dataset.py
```

Für das geplante Minimalmodell werden mindestens zehn Trainingsdateien und zehn
Testdateien pro Klasse empfohlen. Eine Datei darf nur in einer der beiden Mengen
liegen. Vor dem Training müssen Herkunft und Lizenz jeder verwendeten Aufnahme
geklärt sein.

## Synthetischer Prototypdatensatz

Für einen technischen Durchlauf ohne externe Audiodateien kann der lokale
Generator verwendet werden:

```powershell
python scripts/make_demo_dataset.py
```

Die erzeugten WAV-Dateien werden in `metadata.csv` als synthetische,
selbst erzeugte Prototyp-Töne gekennzeichnet. Sie dienen nur dazu, die
Datenprüfung und den Upload-Ablauf zu testen. Für die endgültige Bewertung
werden echte, unterschiedlich aufgenommene und rechtlich geklärte Geräusche
benötigt.
