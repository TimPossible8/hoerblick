const navItems = document.querySelectorAll("[data-screen]");
const screens = document.querySelectorAll(".screen[data-view]");
const audioFile = document.querySelector("#audio-file");
const fileName = document.querySelector("#file-name");
const modelOptions = document.querySelectorAll(".model-option");
const modelHelp = document.querySelector("#model-help");
const resultLabel = document.querySelector("#result-label");
const analyseButton = document.querySelector("#analyse-button");
const resetButton = document.querySelector("#reset-button");

const modelCopy = {
  own: { help: "Mit eigenen Aufnahmen auf die vier Projektklassen trainiert.", label: "Eigenes Modell" },
  yamnet: { help: "Vortrainiertes Modell mit Mapping auf die Projektklassen.", label: "YAMNet" },
};

function showScreen(screenName) {
  screens.forEach((screen) => screen.classList.toggle("is-visible", screen.dataset.view === screenName));
  document.querySelectorAll(".bottom-nav-item").forEach((item) => {
    const isActive = item.dataset.screen === screenName;
    item.classList.toggle("is-active", isActive);
    item.setAttribute("aria-current", isActive ? "page" : "false");
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

navItems.forEach((item) => item.addEventListener("click", () => showScreen(item.dataset.screen)));

audioFile.addEventListener("change", () => {
  const selectedFile = audioFile.files?.[0];
  fileName.textContent = selectedFile ? selectedFile.name : "Noch keine Datei ausgewählt";
});

modelOptions.forEach((option) => {
  option.addEventListener("click", () => {
    const model = option.dataset.model;
    modelOptions.forEach((item) => {
      const isSelected = item === option;
      item.classList.toggle("is-selected", isSelected);
      item.setAttribute("aria-selected", isSelected ? "true" : "false");
    });
    modelHelp.textContent = modelCopy[model].help;
    resultLabel.textContent = `${modelCopy[model].label} · Demo`;
  });
});

analyseButton.addEventListener("click", () => {
  analyseButton.classList.add("is-loading");
  analyseButton.innerHTML = '<span class="material-symbols-rounded">hourglass_top</span> Analyse läuft';
  window.setTimeout(() => {
    analyseButton.classList.remove("is-loading");
    analyseButton.innerHTML = '<span class="material-symbols-rounded">check</span> Analyse abgeschlossen';
    resultLabel.textContent = "Demoergebnis";
  }, 420);
});

resetButton.addEventListener("click", () => {
  resultLabel.textContent = "Demoergebnis";
  analyseButton.innerHTML = '<span class="material-symbols-rounded">graphic_eq</span> Analyse starten';
  audioFile.value = "";
  fileName.textContent = "Noch keine Datei ausgewählt";
});
