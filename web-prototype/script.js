const $ = (selector) => document.querySelector(selector);
const audioFile = $("#audio-file");
const resultPanel = $("#analysis-result");
const analyseButton = $("#analyse-button");
const message = $("#analysis-message");
let selectedModel = "own", statuses = {}, busy = false, requestVersion = 0;
const modelCopy = {
  own: "Eigenes Keras-Modell · vier Projektklassen · 448 Audiomerkmale.",
  yamnet: "YAMNet · 521 AudioSet-Klassen, auf vier Projektklassen abgebildet.",
};
function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("is-error", isError);
}
function invalidateResult() {
  requestVersion += 1;
  resultPanel.hidden = true;
  setMessage("");
}
function updateControls() {
  analyseButton.disabled = busy || !audioFile.files?.length || !statuses[selectedModel]?.ready;
  $("#model-help").textContent = `${modelCopy[selectedModel]} ${statuses[selectedModel]?.message || "Modellstatus wird geprüft …"}`;
  analyseButton.innerHTML = busy
    ? '<span class="material-symbols-rounded" aria-hidden="true">hourglass_top</span> Analyse läuft …'
    : '<span class="material-symbols-rounded" aria-hidden="true">graphic_eq</span> Analyse starten';
}
document.querySelectorAll("[data-screen]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".screen").forEach((screen) => screen.classList.toggle("is-visible", screen.dataset.view === button.dataset.screen));
    document.querySelectorAll(".bottom-nav-item").forEach((item) => {
      const active = item.dataset.screen === button.dataset.screen;
      item.classList.toggle("is-active", active);
      if (active) item.setAttribute("aria-current", "page");
      else item.removeAttribute("aria-current");
    });
    window.scrollTo({ top: 0, behavior: "auto" });
  });
});
audioFile.addEventListener("change", () => {
  invalidateResult();
  $("#file-name").textContent = audioFile.files?.[0]?.name || "Noch keine Datei ausgewählt";
  updateControls();
});
document.querySelectorAll(".model-option").forEach((button) => {
  button.addEventListener("click", () => {
    selectedModel = button.dataset.model;
    invalidateResult();
    document.querySelectorAll(".model-option").forEach((item) => {
      item.classList.toggle("is-selected", item === button);
      item.setAttribute("aria-pressed", String(item === button));
    });
    updateControls();
  });
});
analyseButton.addEventListener("click", async () => {
  const file = audioFile.files?.[0];
  if (!file || busy) return;
  invalidateResult();
  if (file.size > 25 * 1024 * 1024) return setMessage("Die Datei ist zu groß. Maximal 25 MB sind erlaubt.", true);
  const version = requestVersion, model = selectedModel;
  busy = true;
  updateControls();
  setMessage("Die ersten fünf Sekunden werden lokal analysiert …");
  try {
    const response = await fetch(`/api/analyse?model=${model}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
    });
    const result = await response.json();
    if (version !== requestVersion) return;
    if (!response.ok) throw new Error(result.error || "Die Analyse ist fehlgeschlagen.");
    const colors = { Rot: "red", Gelb: "yellow", Grün: "green" };
    const icons = { Rot: "priority_high", Gelb: "info", Grün: "check" };
    resultPanel.className = `warning-panel warning-panel--${colors[result.warning.level]}`;
    $("#warning-symbol").textContent = icons[result.warning.level];
    $("#warning-level-text").textContent = `Warnstufe ${result.warning.level}`;
    $("#result-label").textContent = result.model_name;
    $("#result-title").textContent = result.warning.title;
    $("#result-description").textContent = result.warning.action;
    const percent = Math.round(result.confidence * 100);
    $("#confidence-value").textContent = `${percent} %`;
    $("#confidence-bar").style.width = `${percent}%`;
    $("#result-details").textContent = [
      result.source_label ? `YAMNet-Klasse: ${result.source_label}.` : "Klasse: " + result.label + ".",
      `Analyse: ${Math.round(result.elapsed_ms)} ms.`, `Priorität: ${result.warning.score.toFixed(3)}.`,
    ].join(" ");
    resultPanel.hidden = false;
    setMessage("Analyse abgeschlossen. Modellscore ist keine garantierte Trefferwahrscheinlichkeit.");
    resultPanel.scrollIntoView({ behavior: "auto", block: "center" });
  } catch (error) {
    if (version === requestVersion) setMessage(error.message || "Lokaler Server nicht erreichbar.", true);
  } finally {
    busy = false;
    updateControls();
  }
});
$("#reset-button").addEventListener("click", () => {
  invalidateResult();
  audioFile.value = "";
  $("#file-name").textContent = "Noch keine Datei ausgewählt";
  updateControls();
});
async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Status nicht verfügbar");
    statuses = await response.json();
    const ready = Object.values(statuses).filter((s) => s.ready).length;
    $("#connection-status").textContent = `${ready}/2 Modelle geladen`;
    for (const key of ["own", "yamnet"]) $("#status-" + key).textContent = statuses[key].ready ? "Modell geladen" : "Nicht verfügbar";
    if (!ready) setMessage("Modelle nicht verfügbar. Details unter der Modellwahl.", true);
  } catch {
    $("#connection-status").textContent = "Server nicht erreichbar";
    setMessage("Bitte Start-HoerBlick.ps1 starten und http://127.0.0.1:8765 öffnen.", true);
  }
  updateControls();
}
async function loadComparison() {
  try {
    const response = await fetch("/api/comparison");
    if (!response.ok) throw new Error("Vergleich nicht verfügbar");
    const { rows, metadata } = await response.json();
    const table = $("#comparison-body");
    table.replaceChildren();
    rows.forEach((row) => {
      const element = document.createElement("div");
      element.className = "comparison-row";
      ["Kriterium", "Eigenes Modell", "YAMNet"].forEach((key, index) => {
        const cell = document.createElement(index ? "span" : "strong");
        cell.textContent = row[key];
        element.append(cell);
      });
      table.append(element);
    });
    $("#comparison-note").textContent = metadata.count
      ? `${metadata.count} identische Testdateien · ${metadata.dataset_label}. Kleine Stichprobe; keine allgemeine Erkennungsleistung. Zeit ohne Laden/Aufwärmen.`
      : "Noch kein Vergleichslauf gespeichert.";
  } catch {
    $("#comparison-note").textContent = "Vergleichsdaten nicht erreichbar.";
  }
}
updateControls();
loadStatus();
loadComparison();
