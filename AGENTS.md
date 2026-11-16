# AGENTS.md - HörBlick

## Projektübersicht

HörBlick ist ein schulisches Prototyp-Projekt zur Klausurersatzleistung Informatik. Die Anwendung klassifiziert Umgebungsgeräusche und macht sicherheitsrelevante Schallereignisse (Feueralarm, Sirenen, Türklingeln) für Menschen mit Hörbeeinträchtigung visuell wahrnehmbar.

**Wichtig:** Dies ist ein Bildungsprojekt und Demonstrator - keine zertifizierte Warntechnik oder normgeprüfter Rauchmelder.

## Projektstruktur

- `web-prototype/` - Eigenständige statische HTML/CSS/JS-Oberfläche mit drei Ansichten (Analyse, Vergleich, Über)
- `hoerblick/` - Python/Streamlit-Anwendung mit ML-Modell, Datensätzen und Trainingsskripten
- `docs/screenshots/` - UI-Screenshots und PDF-Dokumentation

## Arbeiten mit diesem Projekt

### Web-Prototyp (web-prototype/)

- Statische HTML-Datei, kann direkt im Browser geöffnet werden
- Verwendet Google Fonts (Roboto) und Material Symbols
- Keine Build-Schritte oder Abhängigkeiten erforderlich
- Mobile-First-Design mit drei Tab-Ansichten
- Bei Änderungen: Konsistenz zwischen allen drei Ansichten prüfen

### Python-Anwendung (hoerblick/)

**Umgebung einrichten:**
```powershell
cd hoerblick
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Optionale ML-Abhängigkeiten:**
- `requirements-ml.txt` enthält TensorFlow, TensorFlow Hub für lokales Training
- Nur installieren, wenn Training oder Modellexport erforderlich ist

**Wichtige Konventionen:**

1. **Datensätze:**
   - Synthetische Testdaten sind eingecheckt (`data/train/`, `data/test/`)
   - Reale Audioaufnahmen bleiben außerhalb von Git
   - Metadaten in `data/metadata.csv` dokumentieren Herkunft und Lizenz

2. **Modelle:**
   - Modellgewichte werden nicht versioniert
   - Modellexporte gehören nach `models/model_a/` (keras_model.h5, labels.txt)
   - Ohne Modell läuft die App im Demo-Modus mit synthetischen Ergebnissen

3. **Klassen:**
   - `alarm` - Feueralarm
   - `siren` - Polizei/Rettungssirenen
   - `doorbell` - Türklingeln
   - `background` - Hintergrundgeräusche

### Testen und Validierung

```powershell
cd hoerblick
python scripts/check_dataset.py          # Datensatzstruktur prüfen
python -m unittest discover -s tests -v  # Unit-Tests ausführen
python scripts/make_demo_dataset.py      # Demo-Datensatz neu erstellen
```

### Dokumentation

Relevante Dokumente im `hoerblick/docs/`:
- `dokumentation.md` - Gesamtprojektdokumentation
- `trainingsprotokoll.md` - Modelltraining und Metriken
- `ki_reflexion.md` - Ethische und gesellschaftliche Überlegungen

## Entwicklungsrichtlinien

### Code-Änderungen

1. **Streamlit-App (`hoerblick/app.py`):**
   - Klare visuelle Hierarchie beibehalten (Farbe, Text, Symbol)
   - Demo-Modus deutlich kennzeichnen
   - Warnungen müssen ohne Audioausgabe verständlich sein

2. **ML-Module (`hoerblick/hoerblick/`):**
   - Audio-Features konsistent halten (Mel-Spektrogramme, MFCCs)
   - YAMNet-Embeddings aus TensorFlow Hub verwenden
   - Modellschnittstelle nicht brechen (Input: 22.05kHz Mono, 1s Segmente)

3. **Skripte (`hoerblick/scripts/`):**
   - Training erwartet `audio-source/` mit Unterordnern: `Feueralarm/`, `sirenen/`, `Klingeln/`, `Hintergrund/`
   - Pfad kann mit `--source` überschrieben werden
   - Metadaten automatisch aktualisieren

### Neue Features

- **Audio-Input:** 16-bit PCM WAV, mono, 22.05 kHz Sample-Rate bevorzugen
- **Accessibility:** Farbschema muss ausreichend Kontrast bieten, Icons benötigen Text-Labels
- **Lizenzierung:** Keine urheberrechtlich geschützten Sounds ohne Klärung verwenden
- **Performance:** Echtzeit-Klassifizierung anstreben (< 100ms Inferenz)

## Dateien, die nicht versioniert werden

- `.venv/` - Virtuelle Python-Umgebung
- `models/*/*.h5` - Trainierte Modellgewichte
- `audio-source/` - Reale Audiodateien für Training
- Persönliche Konfigurationsdateien

## Typische Aufgaben

### Neues Modell trainieren

```powershell
cd hoerblick
python scripts/train_model.py --source audio-source/ --epochs 50
```

### Modell in der App testen

```powershell
streamlit run app.py
```

Im Browser: Audio-Datei hochladen oder Mikrofon verwenden

### Datensatz erweitern

1. Neue WAV-Dateien in `audio-source/<Klasse>/` ablegen
2. `scripts/check_dataset.py` ausführen
3. Metadaten in `data/metadata.csv` aktualisieren (Quelle, Lizenz)
4. Modell neu trainieren

### Web-Prototyp anpassen

1. `web-prototype/index.html` bearbeiten
2. Im Browser öffnen und alle drei Tabs prüfen
3. Mobile Ansicht testen (responsive Design)

## Technische Einschränkungen

- **Modellgröße:** Teachable Machine-Export, limitiert auf einfache CNN-Architekturen
- **Audioformat:** WAV bevorzugt, MP3/OGG benötigen zusätzliche Konvertierung
- **Browser-Kompatibilität:** Web-Prototyp benötigt moderne Browser (ES6+)
- **Python-Version:** Mindestens 3.11 erforderlich

## Kontextinformationen für Agents

- **Zielgruppe:** Lehrkräfte und Schüler im Informatikunterricht, Projekt im Rahmen einer Klausurersatzleistung
- **Zweck:** Demonstration von Audio-ML, UI/UX für Accessibility, ethische KI-Reflexion
- **Nicht-Ziele:** Produktionsreife, echte Sicherheitstechnik, kommerzielle Nutzung
- **Bewertungsfokus:** Technische Umsetzung, Dokumentation, Reflexion gesellschaftlicher Auswirkungen
