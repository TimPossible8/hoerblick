# AGENTS.md - HörBlick Python-Anwendung

## Überblick

Dieser Ordner enthält die Python/Streamlit-Implementierung von HörBlick - eine Audio-Klassifizierungsanwendung für sicherheitsrelevante Geräusche. Die Anwendung verwendet maschinelles Lernen, um Feueralarm, Sirenen, Türklingeln und Hintergrundgeräusche zu identifizieren.

## Ordnerstruktur

```
hoerblick/
├── app.py                    # Haupt-Streamlit-Anwendung
├── requirements.txt          # Standard-Abhängigkeiten
├── requirements-ml.txt       # Optionale ML-Abhängigkeiten (TensorFlow)
├── data/                     # Trainings- und Testdatensätze
│   ├── metadata.csv          # Herkunft und Lizenz jeder Audiodatei
│   ├── train/                # Trainingsdaten (4 Klassen)
│   └── test/                 # Testdaten (4 Klassen)
├── docs/                     # Projektdokumentation
│   ├── dokumentation.md
│   ├── trainingsprotokoll.md
│   └── ki_reflexion.md
├── hoerblick/                # Python-Modul mit Core-Logik
│   ├── __init__.py
│   ├── audio_processor.py    # Audio-Vorverarbeitung
│   ├── classifier.py         # Modellklassifizierung
│   └── feature_extractor.py  # Feature-Extraktion (MFCCs, Mel-Spektrogramme)
├── models/                   # Modellexporte (nicht versioniert)
│   ├── model_a/              # Teachable Machine Export
│   │   ├── keras_model.h5    # Trainierte Gewichte
│   │   ├── labels.txt        # Klassenlabels
│   │   └── metrics.json      # Testmetriken
│   └── model_b/              # Alternative Modelle
├── scripts/                  # Utility-Skripte
│   ├── check_dataset.py      # Datensatzvalidierung
│   ├── make_demo_dataset.py  # Demo-Daten generieren
│   └── train_model.py        # Modelltraining
└── tests/                    # Unit-Tests
    ├── test_audio_processor.py
    ├── test_classifier.py
    └── test_feature_extractor.py
```

## Setup und Ausführung

### Virtuelle Umgebung einrichten

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Streamlit-App starten

```powershell
streamlit run app.py
```

Die App öffnet sich automatisch im Browser auf `http://localhost:8501`.

### ML-Abhängigkeiten (optional)

Nur erforderlich für Training oder Modellexport:

```powershell
pip install -r requirements-ml.txt
```

## Wichtige Konventionen

### Audio-Format

- **Sample-Rate:** 22.05 kHz (fest)
- **Kanäle:** Mono (1 Kanal)
- **Format:** 16-bit PCM WAV bevorzugt
- **Segmentlänge:** 1 Sekunde für Klassifizierung

### Klassen

Die Anwendung klassifiziert vier Kategorien:

1. **alarm** - Feueralarm, Rauchmelder
2. **siren** - Polizei-, Rettungs-, Feuerwehrsirenen
3. **doorbell** - Türklingeln, Gongs
4. **background** - Hintergrundgeräusche, Sprache, Musik

### Datensätze

**Synthetische Daten (eingecheckt):**
- 48 Trainingsdateien in `data/train/`
- 40 Testdateien in `data/test/`
- Dienen zur Prüfung der Ordnerstruktur und des Ablaufs
- **Nicht** für produktive Modellbewertung geeignet

**Reale Daten (außerhalb Git):**
- Quellordner: `audio-source/` im Repository-Root
- Unterordner: `Feueralarm/`, `sirenen/`, `Klingeln/`, `Hintergrund/`
- Metadaten werden automatisch in `data/metadata.csv` dokumentiert

### Modelle

**Modellexport-Struktur:**
```
models/model_a/
├── keras_model.h5    # TensorFlow/Keras-Modell
├── labels.txt        # Klassenreihenfolge (eine pro Zeile)
└── metrics.json      # Evaluation-Metriken (Accuracy, Precision, Recall, F1)
```

**Demo-Modus:**
- Ohne `keras_model.h5` läuft die App im Demo-Modus
- Zeigt synthetische Klassifizierungsergebnisse
- Wird in der UI deutlich gekennzeichnet

## Entwicklungsworkflow

### Datensatz prüfen

```powershell
python scripts/check_dataset.py
```

Validiert:
- Ordnerstruktur (`train/` und `test/` mit je 4 Klassen)
- Dateiformat (WAV, 22.05 kHz, Mono)
- Metadaten-Vollständigkeit

### Tests ausführen

```powershell
# Alle Tests
python -m unittest discover -s tests -v

# Einzelner Test
python -m unittest tests.test_classifier -v
```

### Modell trainieren

```powershell
python scripts/train_model.py --source audio-source/ --epochs 50 --batch-size 32
```

Optionen:
- `--source`: Pfad zum Quellordner mit realen Audioaufnahmen
- `--epochs`: Anzahl Trainingsepochen (Standard: 50)
- `--batch-size`: Batch-Größe (Standard: 32)
- `--output`: Zielordner für Modellexport (Standard: `models/model_a/`)

### Demo-Datensatz neu erstellen

```powershell
python scripts/make_demo_dataset.py
```

Generiert synthetische Testdaten mit NumPy und SciPy.

## Streamlit-App Änderungen

### UI-Komponenten

Die App hat drei Hauptbereiche:

1. **Datei-Upload / Mikrofon-Input**
   - `st.file_uploader()` für WAV-Dateien
   - `st.audio()` für Wiedergabe
   - Audio-Input-Widget (experimentell)

2. **Klassifizierungsergebnis**
   - Farb-codierte Warnungen (Rot für Alarm/Siren, Gelb für Doorbell, Grün für Background)
   - Icon + Text + Konfidenzbalken
   - Accessibility-freundlich (funktioniert ohne Farbe)

3. **Feature-Visualisierung**
   - Mel-Spektrogramm mit Matplotlib
   - MFCC-Koeffizienten
   - Optional: Waveform-Plot

### Wichtige Dateien

- **app.py:** Streamlit-UI-Logik, Page-Layout, Session-State
- **hoerblick/classifier.py:** Modell laden, Inferenz ausführen
- **hoerblick/audio_processor.py:** Resampling, Mono-Konvertierung, Normalisierung
- **hoerblick/feature_extractor.py:** Mel-Spektrogramme, MFCCs, YAMNet-Embeddings

### Styling

Custom CSS in `app.py`:
- Warnfarben konsistent halten
- Material Design Icons (optional)
- Responsive Layout für mobile Geräte

## Modellarchitektur

### Teachable Machine Export

- Einfache CNN-Architektur (TensorFlow)
- Input: Mel-Spektrogramm (128 Bins x ~44 Frames)
- Output: Softmax über 4 Klassen
- Begrenzte Anpassbarkeit, aber einfach zu trainieren

### Alternative Ansätze

Für bessere Ergebnisse:
- YAMNet Transfer Learning (TensorFlow Hub)
- Custom CNN mit Data Augmentation
- Ensemble aus mehreren Modellen

## Testing

### Unit-Tests

- **test_audio_processor.py:** Resampling, Formatkonvertierung
- **test_classifier.py:** Modellschnittstelle, Demo-Modus
- **test_feature_extractor.py:** MFCC-Dimensionen, Spektrogramm-Shape

### Integration-Tests

- End-to-End-Klassifizierung mit Beispieldateien
- UI-Tests mit Streamlit Test-Framework (falls verfügbar)

### Manuelle Tests

1. App starten mit und ohne Modell
2. WAV-Datei hochladen (verschiedene Sample-Rates)
3. Mikrofon-Input testen (Chrome/Firefox)
4. Mobile Ansicht prüfen

## Fehlerbehebung

### Häufige Probleme

**Modell nicht gefunden:**
- Prüfen, ob `models/model_a/keras_model.h5` existiert
- Falls nicht vorhanden: Training ausführen oder Demo-Modus akzeptieren

**Audio-Format-Fehler:**
- Nur WAV mit 22.05 kHz Mono wird direkt unterstützt
- Andere Formate/Sample-Rates werden automatisch konvertiert (benötigt librosa)

**Streamlit langsam:**
- Cache für Feature-Extraktion nutzen: `@st.cache_data`
- Modell nur einmal laden: `@st.cache_resource`

**Tests schlagen fehl:**
```powershell
# Abhängigkeiten neu installieren
pip install --upgrade -r requirements.txt

# Cache löschen
python -m pytest --cache-clear
```

## Performance-Überlegungen

### Echtzeit-Klassifizierung

- Inferenz-Zeit: < 100ms auf CPU anstreben
- Mel-Spektrogramm-Berechnung: ~20-50ms
- Modellgröße: < 10 MB für schnelles Laden

### Optimierungen

- TensorFlow Lite für Mobile/Edge-Deployment
- Quantisierung auf INT8 (mit Qualitätsverlust)
- Batching für mehrere Audiofragmente

## Lizenz und Datenschutz

- **Synthetische Daten:** Frei verwendbar
- **Reale Audioaufnahmen:** Lizenz vor Verwendung klären
- **Trainingsdaten:** Keine persönlichen Sprachaufnahmen ohne Einwilligung
- **Modellgewichte:** Nicht versioniert, lokal gespeichert

## Nächste Schritte

Für Weiterentwicklung:

1. **Datensammlung:** Reale, lizenzierte Audioaufnahmen für jede Klasse
2. **Data Augmentation:** Zeit-Stretching, Pitch-Shifting, Rauschen
3. **Modellverbesserung:** Transfer Learning mit YAMNet oder VGGish
4. **Evaluation:** Confusion Matrix, Per-Class-Metriken, Cross-Validation
5. **Deployment:** Web-API (FastAPI), Mobile App (TensorFlow Lite)

## Kontakt

Dies ist ein schulisches Projekt. Für Fragen zur Implementierung siehe Dokumentation in `docs/`.
