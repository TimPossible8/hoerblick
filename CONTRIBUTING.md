# Mitwirken

Dieses Repository dokumentiert eine schulische Klausurersatzleistung. Änderungen sollen den Prototyp verständlich halten und die Trennung zwischen Trainings- und Testdaten bewahren.

Nach Änderungen an Python-Code aus dem Ordner `hoerblick` ausführen:

```powershell
python -m unittest discover -s tests -v
python scripts/check_dataset.py
```

Neue Audiodateien werden unter `hoerblick/data/train/<klasse>/` oder `hoerblick/data/test/<klasse>/` abgelegt und in `hoerblick/data/metadata.csv` mit Quelle und Lizenz dokumentiert. Eine Datei gehört nur zu einem der beiden Datensätze. Mitgelieferte Modellgewichte, virtuelle Umgebungen und Zugangsdaten gehören nicht ins Repository.
