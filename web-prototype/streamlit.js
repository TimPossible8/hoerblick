// Streamlit's component protocol; the same assets still work with serve.py.
if (new URLSearchParams(window.location.search).has("streamlitUrl")) {
  const parentOrigin = new URL(new URLSearchParams(window.location.search).get("streamlitUrl")).origin;
  const send = (type, payload = {}) => window.parent.postMessage({ isStreamlitMessage: true, type, ...payload }, parentOrigin);
  let ready;
  const rendered = new Promise((resolve) => { ready = resolve; });
  const pending = new Map();
  let sequence = 0;
  window.addEventListener("message", (event) => {
    if (event.source !== window.parent || event.origin !== parentOrigin || event.data?.type !== "streamlit:render") return;
    ready();
    const reply = event.data.args?.response;
    const task = pending.get(reply?.id);
    if (!task) return;
    pending.delete(reply.id);
    clearTimeout(task.timer);
    // Clear the component's widget value so raw audio is not kept in session state.
    send("streamlit:setComponentValue", { value: null, dataType: "json" });
    if (reply.error) task.reject(new Error(reply.error));
    else task.resolve(reply.data);
  });
  async function request(action, payload = {}) {
    await rendered;
    const id = `${Date.now()}-${++sequence}`;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        send("streamlit:setComponentValue", { value: null, dataType: "json" });
        reject(new Error("Die Serverantwort dauert zu lange. Bitte die Seite neu laden."));
      }, 300000);
      pending.set(id, { resolve, reject, timer });
      send("streamlit:setComponentValue", { value: { id, action, ...payload }, dataType: "json" });
    });
  }
  const initial = request("init");
  window.hoerblickTransport = {
    status: async () => (await initial).statuses,
    comparison: async () => (await initial).comparison,
    analyse: async (file, model) => {
      const audio = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(",")[1]);
        reader.onerror = () => reject(new Error("Die Datei konnte nicht gelesen werden."));
        reader.readAsDataURL(file);
      });
      return request("analyse", { model, audio });
    },
  };
  document.querySelector(".eyebrow").textContent = "ONLINE-PROTOTYP";
  document.querySelector(".input-hint").lastChild.textContent = " Analyse auf dem Server · erste 5 Sekunden · kein Speichern des Uploads.";
  document.querySelector("#privacy-details").textContent = "Die Datei wird zur Analyse an den Streamlit-Server übertragen und von der App nicht als Datei gespeichert. YAMNet wird beim ersten Start heruntergeladen. Google Fonts lädt Schriftarten aus dem Internet.";
  initial.then(({ csv }) => {
    const url = URL.createObjectURL(new Blob(["\ufeff", csv], { type: "text/csv;charset=utf-8" }));
    document.querySelector("#comparison-download").href = url;
    window.addEventListener("pagehide", () => URL.revokeObjectURL(url), { once: true });
  }).catch(() => {});
  document.documentElement.classList.add("streamlit-host");
  const resize = () => {
    let height = 900;
    try { height = window.parent.innerHeight; } catch { /* Cross-origin host fallback. */ }
    send("streamlit:setFrameHeight", { height });
  };
  window.addEventListener("resize", resize);
  send("streamlit:componentReady", { apiVersion: 1 });
  resize();
}
