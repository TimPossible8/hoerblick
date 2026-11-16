$ErrorActionPreference = "Stop"
$projectDirectory = Join-Path $PSScriptRoot "hoerblick"
$pythonPath = Join-Path $projectDirectory ".venv-ml312\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = Join-Path $projectDirectory ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python-Umgebung fehlt. Bitte zuerst die Installation aus hoerblick/README.md durchführen."
}
Write-Host "HörBlick öffnen: http://127.0.0.1:8765"
& $pythonPath (Join-Path $projectDirectory "serve.py")
