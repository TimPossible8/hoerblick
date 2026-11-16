# HörBlick Dokumentation

## 1 Problem und Zielgruppe

Viele Warnsignale im Alltag funktionieren ausschließlich über Schall. HörBlick soll solche Ereignisse sichtbar und verständlich machen. Die zentrale Persona ist Lena, 19, hochgradig schwerhörig und auf dem Weg in die erste eigene Wohnung.

## 2 Funktionsumfang

Der Minimalprototyp analysiert eine hochgeladene WAV-, MP3- oder FLAC-Datei. Die Nutzerin oder der Nutzer wählt zwischen dem eigenen Modell und YAMNet. Anschließend zeigt die App die erkannte Klasse, die Konfidenz und eine Warnstufe.

## 3 Prioritätsalgorithmus

Der Prioritätswert wird transparent aus Gefahrenwert, Dringlichkeit und Modellkonfidenz gebildet:

```text
Priorität = Gefahrenwert * Dringlichkeitswert * Konfidenz
```

Feueralarm und Sirene erhalten hohe Werte. Eine Türklingel wird als gelbe Warnung behandelt. Unsichere Vorhersagen unter Score 0,35 werden gelb mit Prüfhinweis ausgegeben. Nur ausreichend sicher zugeordneter Hintergrund wird grün angezeigt; Grün ist keine Entwarnung.

## 4 Modelle

Version A ist ein lokal trainierter ExtraTrees-Klassifikator auf zusammengefassten Log-Mel-Audiomerkmalen mit trainierbarem Keras-Kalibrierungskopf. Das Modell wird als Keras-HDF5 exportiert. Die aktuelle Validierungsgenauigkeit beträgt 87,5 % (7/8), die unabhängige Testgenauigkeit 66,7 % (8/12). Version B nutzt YAMNet aus TensorFlow Hub. Der kleine Datensatz macht beide Messwerte unsicher; die Validierungsleistung ist kein Nachweis für dieselbe Leistung bei neuen Aufnahmen.

## 5 Grenzen

Die App ersetzt keine zertifizierte Warntechnik. Geräusche aus großer Entfernung, starke Hintergrundgeräusche und seltene Klangvarianten können zu Fehlklassifikationen führen.

## 6 Gemeinsame Messwerte

Die Kennzahlen des gemeinsamen Laufs stehen nach Ausführung des Vergleichsskripts in `vergleich.csv`. YAMNet wird vortrainiert verwendet.

## Ergänzung nach Schulstunde 3

Die technische Integration ist in [schulstunde_3.md](schulstunde_3.md)
dokumentiert. Das fertige Web-Design startet über `python serve.py`.
Keras-Modell und YAMNet verwenden reale Inferenz; die Streamlit-Alternative
teilt dieselben Adapter. Aktuelle Messergebnisse werden ausschließlich aus
`../vergleich.csv`, `testprotokoll.csv` und `vergleich_lauf.json` übernommen.
