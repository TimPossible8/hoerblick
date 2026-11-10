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

Feueralarm und Sirene erhalten hohe Werte. Eine Türklingel wird als gelbe Warnung behandelt. Hintergrundgeräusche und sehr unsichere Vorhersagen werden grün ausgegeben.

## 4 Modelle

Version A wird mit Teachable Machine auf eigenen Trainingsdaten trainiert. Version B nutzt YAMNet aus TensorFlow Hub. Beide Varianten werden mit demselben getrennten Testdatensatz geprüft.

## 5 Grenzen

Die App ersetzt keine zertifizierte Warntechnik. Geräusche aus großer Entfernung, starke Hintergrundgeräusche und seltene Klangvarianten können zu Fehlklassifikationen führen.

## 6 Offene Messwerte

Die Kennzahlen werden in `vergleich.csv` ergänzt, sobald beide Modelle trainiert und unter denselben Bedingungen getestet wurden.
