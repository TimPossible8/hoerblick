// Run with: node --test web-prototype/streamlit.test.cjs
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

test("Streamlit bridge matches replies, ignores foreign messages and clears uploads", async () => {
  const messages = [], handlers = {}, elements = new Map();
  const parent = { innerHeight: 812, postMessage: (message) => messages.push(message) };
  const window = {
    parent, location: { search: "?streamlitUrl=https%3A%2F%2Fexample.streamlit.app%2F" },
    addEventListener: (event, handler) => { handlers[event] = handler; },
  };
  const context = vm.createContext({
    window, URL, URLSearchParams, Blob, setTimeout, clearTimeout,
    document: {
      querySelector: (selector) => {
        if (!elements.has(selector)) elements.set(selector, { lastChild: {} });
        return elements.get(selector);
      },
      documentElement: { classList: { add() {} } },
    },
    FileReader: class {
      readAsDataURL() { this.result = "data:audio/wav;base64,YXVkaW8="; this.onload(); }
    },
  });
  vm.runInContext(fs.readFileSync(__dirname + "/streamlit.js", "utf8"), context);
  const render = (response, origin = "https://example.streamlit.app") =>
    handlers.message({ source: parent, origin, data: { type: "streamlit:render", args: { response } } });
  render();
  await new Promise(setImmediate);
  const initial = messages.find((message) => message.value?.action === "init");
  assert.ok(initial);
  const statuses = { own: { ready: false }, yamnet: { ready: true } };
  render({ id: initial.value.id, data: { statuses, comparison: { rows: [] }, csv: "Kriterium" } });
  assert.equal((await window.hoerblickTransport.status()).yamnet.ready, true);

  const resultPromise = window.hoerblickTransport.analyse({}, "yamnet");
  await new Promise(setImmediate);
  const request = messages.find((message) => message.value?.action === "analyse");
  assert.equal(request.value.audio, "YXVkaW8=");
  let resolved = false;
  resultPromise.then(() => { resolved = true; });
  render({ id: request.value.id, data: { label: "forged" } }, "https://other.example");
  render({ id: "stale-request", data: { label: "stale" } });
  await new Promise(setImmediate);
  assert.equal(resolved, false);
  render({ id: request.value.id, data: { label: "alarm" } });
  assert.equal((await resultPromise).label, "alarm");
  assert.equal(messages.at(-1).value, null);
  assert.ok(messages.some((message) => message.type === "streamlit:setFrameHeight" && message.height === 812));
});
