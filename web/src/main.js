/**
 * Signal — browser entry point.
 *
 * Pipeline: getUserMedia -> MediaPipe HandLandmarker (WASM) -> feature vector
 * -> exported MLP -> hold/commit state machine -> captions, speech, and the
 * presenter window. Every stage runs in the tab; no frame leaves the device.
 */

import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

import {
  GestureClassifier,
  labelToText,
  categoryFor,
  MODE_ALL,
  MODE_WORDS,
  MODE_LETTERS,
} from "./classifier.js?v=2";
import { SpellCorrector, WordProcessor } from "./words.js?v=2";
import { Speaker } from "./speech.js?v=2";
import { groupLabels, shapeFor, MOTION_LETTERS, CONFUSABLE } from "./guide.js?v=2";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const LANDMARKER_TASK =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const CHANNEL = "signal-captions";
const HOLD_CIRCUMFERENCE = 128.8;
const STORAGE_KEY = "signal.settings.v1";
const COOLDOWN_MS = 600;

const MODE_HINTS = {
  [MODE_ALL]: "Everything at once. Switch to Letters when fingerspelling a name.",
  [MODE_WORDS]: "Only whole words and phrases — fastest for conversation.",
  [MODE_LETTERS]: "Only A–Z and 0–9. Best accuracy for spelling out names.",
};
const MODE_PILLS = {
  [MODE_ALL]: "All gestures",
  [MODE_WORDS]: "Words only",
  [MODE_LETTERS]: "Letters only",
};
const MODE_ORDER = [MODE_ALL, MODE_WORDS, MODE_LETTERS];

const el = (id) => document.getElementById(id);
const dom = Object.fromEntries(
  [
    "viewport", "video", "overlay", "onboard", "onboard-note", "start-btn",
    "hud", "fps-pill", "mode-pill", "readout", "readout-label", "readout-pct",
    "readout-meter", "hold", "hold-fill", "hold-text",
    "caption", "caption-sub", "correction", "speak-all-btn",
    "history", "download-btn",
    "mode-hint", "hold-range", "hold-out", "pause-range", "pause-out",
    "speak-toggle", "voice-fields", "voice-select", "rate-range", "rate-out",
    "voice-hint",
    "space-btn", "undo-btn", "back-btn", "copy-btn", "clear-btn", "present-btn",
    "camera-field", "camera-select", "skeleton-toggle", "mirror-toggle",
    "status-model", "status-labels", "status-camera",
    "guide-btn", "guide-dialog", "guide-close", "guide-search", "guide-groups",
    "theme-btn", "toast",
  ].map((id) => [id.replace(/-([a-z])/g, (_, c) => c.toUpperCase()), el(id)]),
);

/* ------------------------------------------------------------- settings */

const defaults = {
  theme: null,
  mode: MODE_ALL,
  holdMs: 330,
  pauseMs: 730,
  speak: false,
  voiceURI: null,
  rate: 1,
  skeleton: true,
  mirror: true,
  cameraId: null,
};

function loadSettings() {
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") };
  } catch {
    return { ...defaults };
  }
}

function saveSettings() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    /* Private browsing or blocked storage — preferences just won't persist. */
  }
}

const settings = loadSettings();

const state = {
  landmarker: null,
  classifier: null,
  words: null,
  speaker: new Speaker(),
  stream: null,
  running: false,
  currentLabel: null,
  holdStart: 0,
  cooldownUntil: 0,
  lastCommitted: "",
  lastVideoTime: -1,
  frameTimes: [],
  channel: null,
  presenter: null,
  entries: [],
};

/* -------------------------------------------------------------- helpers */

let toastTimer = null;
function toast(message) {
  dom.toast.textContent = message;
  dom.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    dom.toast.hidden = true;
  }, 2400);
}

function setStatus(node, text, kind = "") {
  node.textContent = text;
  node.className = kind;
}

function applyTheme() {
  const root = document.documentElement;
  if (settings.theme) root.dataset.theme = settings.theme;
  else delete root.dataset.theme;
}

/* ----------------------------------------------------------- load model */

async function loadAssets() {
  setStatus(dom.statusModel, "loading…");
  const [vision, classifier, corrector] = await Promise.all([
    FilesetResolver.forVisionTasks(WASM_BASE),
    GestureClassifier.load("./public/model/gesture_mlp.json"),
    SpellCorrector.load("./public/model/words.json"),
  ]);

  state.landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: LANDMARKER_TASK, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.7,
    minTrackingConfidence: 0.6,
  });

  state.classifier = classifier;
  classifier.setMode(settings.mode);
  state.words = new WordProcessor(corrector, { pauseMs: settings.pauseMs });

  setStatus(dom.statusModel, "ready", "ok");
  setStatus(dom.statusLabels, `${classifier.labels.length} gestures`);
  buildGuide(classifier.labels);
  render();
}

/* ---------------------------------------------------------------- guide */

let guideGroups = [];

function buildGuide(labels) {
  guideGroups = groupLabels(labels).map((group) => ({
    title: group.title,
    items: group.items.map((label) => ({
      label,
      text: labelToText(label),
      shape: shapeFor(label),
      motion: MOTION_LETTERS.has(label),
      confusable: CONFUSABLE.has(label),
    })),
  }));
  renderGuide("");
}

function flagEl(text, title) {
  const flag = document.createElement("span");
  flag.className = "guide-flag";
  flag.textContent = text;
  flag.title = title;
  return flag;
}

function renderGuide(query) {
  const q = query.trim().toLowerCase();
  dom.guideGroups.replaceChildren();
  let shown = 0;

  for (const group of guideGroups) {
    const items = group.items.filter(
      (item) =>
        !q ||
        item.label.toLowerCase().includes(q) ||
        item.text.toLowerCase().includes(q) ||
        (item.shape && item.shape.toLowerCase().includes(q)),
    );
    if (!items.length) continue;
    shown += items.length;

    const section = document.createElement("section");
    section.className = "guide-group";
    const heading = document.createElement("h3");
    heading.textContent = `${group.title} · ${items.length}`;
    const list = document.createElement("div");
    list.className = "guide-list";

    for (const item of items) {
      const row = document.createElement("div");
      row.className = "guide-item";

      const key = document.createElement("span");
      key.className = "guide-key";
      key.textContent = item.label.length === 1 ? item.label : item.text;

      const desc = document.createElement("span");
      desc.className = "guide-desc";
      desc.textContent = item.shape || item.text;
      if (item.motion) {
        desc.append(
          flagEl("motion", "Needs movement — the static model cannot distinguish this yet."),
        );
      }
      if (item.confusable) {
        desc.append(
          flagEl("similar", "Differs from a nearby gesture by one finger — hold it clearly."),
        );
      }

      row.append(key, desc);
      list.append(row);
    }
    section.append(heading, list);
    dom.guideGroups.append(section);
  }

  if (!shown) {
    const empty = document.createElement("p");
    empty.className = "guide-none";
    empty.textContent = `Nothing matches “${query}”.`;
    dom.guideGroups.append(empty);
  }
}

/* --------------------------------------------------------------- camera */

async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    if (cams.length < 2) {
      dom.cameraField.hidden = true;
      return;
    }
    dom.cameraSelect.replaceChildren();
    cams.forEach((cam, i) => {
      const option = document.createElement("option");
      option.value = cam.deviceId;
      option.textContent = cam.label || `Camera ${i + 1}`;
      dom.cameraSelect.append(option);
    });
    if (settings.cameraId) dom.cameraSelect.value = settings.cameraId;
    dom.cameraField.hidden = false;
  } catch {
    dom.cameraField.hidden = true;
  }
}

async function startCamera() {
  dom.startBtn.disabled = true;
  dom.startBtn.textContent = "Starting…";

  try {
    if (!state.landmarker) await loadAssets();

    state.stream?.getTracks().forEach((t) => t.stop());
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: settings.cameraId
        ? { deviceId: { exact: settings.cameraId } }
        : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });

    dom.video.srcObject = state.stream;
    await dom.video.play();
    dom.overlay.width = dom.video.videoWidth;
    dom.overlay.height = dom.video.videoHeight;

    dom.viewport.classList.add("live");
    dom.hud.hidden = false;
    dom.readout.hidden = false;
    setStatus(dom.statusCamera, "live", "ok");
    listCameras();

    if (!state.running) {
      state.running = true;
      requestAnimationFrame(loop);
    }
  } catch (error) {
    dom.startBtn.disabled = false;
    dom.startBtn.textContent = "Enable camera";
    dom.onboardNote.textContent =
      error?.name === "NotAllowedError"
        ? "Camera permission was denied. Allow it in your browser's site settings, then try again."
        : `Could not start the camera: ${error.message}`;
    setStatus(dom.statusCamera, "unavailable", "bad");
  }
}

/* ------------------------------------------------------------ draw/loop */

function drawSkeleton(ctx, landmarks, w, h) {
  ctx.lineWidth = Math.max(2, w / 330);
  ctx.strokeStyle = "rgba(233, 178, 122, 0.9)";
  ctx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(253, 248, 240, 0.95)";
  const r = Math.max(2.5, w / 270);
  for (const point of landmarks) {
    ctx.beginPath();
    ctx.arc(point.x * w, point.y * h, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function loop() {
  if (!state.running) return;
  const now = performance.now();
  const { video, overlay } = dom;

  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;

    state.frameTimes.push(now);
    while (state.frameTimes.length && now - state.frameTimes[0] > 1000) {
      state.frameTimes.shift();
    }
    dom.fpsPill.textContent = `${state.frameTimes.length} fps`;

    const result = state.landmarker.detectForVideo(video, now);
    const landmarks = result.landmarks?.[0] || null;
    const handedness = result.handednesses?.[0]?.[0]?.categoryName || "Right";

    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (landmarks && settings.skeleton) {
      drawSkeleton(ctx, landmarks, overlay.width, overlay.height);
    }

    step(landmarks, handedness, now);
  }

  requestAnimationFrame(loop);
}

/**
 * Hold/commit state machine. A gesture must stay stable for holdMs before it is
 * written, then the same gesture is suppressed for a cooldown so a resting hand
 * does not repeat it.
 */
function step(landmarks, handedness, now) {
  const { classifier, words } = state;
  const { label, confidence } = classifier.recognize(
    landmarks,
    handedness,
    dom.overlay.width,
    dom.overlay.height,
  );
  const text = labelToText(label);

  if (!label) {
    state.currentLabel = null;
    state.holdStart = 0;
    words.clearPreview();
    if (words.onIdle(now)) {
      state.lastCommitted = "";
      commitFeedback(words.lastCommitted);
    }
    dom.readoutLabel.textContent = landmarks ? "Reading…" : "No hand detected";
    dom.readoutPct.textContent = "";
    dom.readoutMeter.style.width = "0%";
    dom.hold.hidden = true;
    render();
    return;
  }

  dom.readoutLabel.textContent = text;
  dom.readoutPct.textContent = `${Math.round(confidence * 100)}%`;
  dom.readoutMeter.style.width = `${Math.round(confidence * 100)}%`;
  words.setPreview(text);

  if (label !== state.currentLabel) {
    state.currentLabel = label;
    state.holdStart = now;
  }

  const progress = Math.min(1, (now - state.holdStart) / settings.holdMs);
  dom.hold.hidden = false;
  dom.holdText.textContent = text;
  dom.holdFill.style.strokeDashoffset = String(HOLD_CIRCUMFERENCE * (1 - progress));

  if (progress >= 1 && now >= state.cooldownUntil && text !== state.lastCommitted) {
    words.addGesture(text, categoryFor(label));
    state.lastCommitted = text;
    state.cooldownUntil = now + COOLDOWN_MS;
    state.holdStart = now;
    words.clearPreview();
    dom.hold.hidden = true;
    commitFeedback(words.lastCommitted);
  }

  render();
}

/** Visual pulse, spoken word, and log entry for a newly committed word. */
function commitFeedback(word) {
  dom.viewport.classList.remove("committed");
  void dom.viewport.offsetWidth; // restart the animation
  dom.viewport.classList.add("committed");

  if (!word) return;
  state.speaker.speak(word);
  addHistory(word);
}

function addHistory(word) {
  const time = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  state.entries.push({ word, time });

  if (state.entries.length === 1) dom.history.replaceChildren();
  const li = document.createElement("li");
  const w = document.createElement("span");
  w.className = "history-word";
  w.textContent = word;
  const t = document.createElement("span");
  t.className = "history-time";
  t.textContent = time;
  li.append(w, t);
  dom.history.append(li);
  dom.history.scrollTop = dom.history.scrollHeight;
}

/* --------------------------------------------------------------- render */

function render() {
  const words = state.words;
  if (!words) return;

  const committed = words.transcript;
  const buffer = words.buffer;

  dom.caption.replaceChildren();
  if (!committed && !buffer) {
    const empty = document.createElement("span");
    empty.className = "caption-empty";
    empty.textContent = "Your signed words appear here…";
    dom.caption.append(empty);
  } else {
    if (committed) dom.caption.append(document.createTextNode(committed));
    if (buffer) {
      if (committed && !committed.endsWith(" ")) {
        dom.caption.append(document.createTextNode(" "));
      }
      const span = document.createElement("span");
      span.className = "caption-live";
      span.textContent = buffer;
      dom.caption.append(span);
    }
  }

  dom.captionSub.textContent = words.preview ? `Holding “${words.preview}”` : " ";
  dom.correction.textContent = words.lastCorrection
    ? `corrected → ${words.lastCorrection}`
    : "";
  dom.undoBtn.disabled = !words.canUndo;

  state.channel?.postMessage({ text: committed, live: buffer });
}

/* ------------------------------------------------------------- controls */

function setMode(mode) {
  settings.mode = mode;
  saveSettings();
  state.classifier?.setMode(mode);
  dom.modePill.textContent = MODE_PILLS[mode];
  dom.modeHint.textContent = MODE_HINTS[mode];
  document.querySelectorAll(".seg").forEach((btn) => {
    const on = btn.dataset.mode === mode;
    btn.classList.toggle("on", on);
    btn.setAttribute("aria-checked", String(on));
  });
}

function populateVoices(voices) {
  if (!voices.length) {
    dom.voiceHint.hidden = false;
    dom.voiceFields.hidden = true;
    return;
  }
  dom.voiceHint.hidden = true;
  dom.voiceSelect.replaceChildren();
  for (const voice of voices) {
    const option = document.createElement("option");
    option.value = voice.voiceURI;
    option.textContent = `${voice.name} (${voice.lang})`;
    dom.voiceSelect.append(option);
  }
  if (settings.voiceURI && voices.some((v) => v.voiceURI === settings.voiceURI)) {
    dom.voiceSelect.value = settings.voiceURI;
    state.speaker.voiceURI = settings.voiceURI;
  }
}

function applySettingsToUI() {
  applyTheme();
  setMode(settings.mode);

  dom.holdRange.value = settings.holdMs;
  dom.holdOut.textContent = `${(settings.holdMs / 1000).toFixed(2)}s`;
  dom.pauseRange.value = settings.pauseMs;
  dom.pauseOut.textContent = `${(settings.pauseMs / 1000).toFixed(2)}s`;

  dom.skeletonToggle.checked = settings.skeleton;
  dom.mirrorToggle.checked = settings.mirror;
  dom.viewport.classList.toggle("mirrored", settings.mirror);

  dom.speakToggle.checked = settings.speak;
  dom.voiceFields.hidden = !settings.speak;
  dom.rateRange.value = settings.rate;
  dom.rateOut.textContent = `${Number(settings.rate).toFixed(1)}×`;

  state.speaker.enabled = settings.speak;
  state.speaker.rate = settings.rate;
  state.speaker.voiceURI = settings.voiceURI;
  state.speaker.onVoicesChanged = populateVoices;
  populateVoices(state.speaker.voices);
  if (!Speaker.supported) {
    dom.speakToggle.disabled = true;
    dom.voiceHint.hidden = false;
    dom.voiceHint.textContent = "This browser does not support speech synthesis.";
  }
}

function transcriptText() {
  return state.words?.displayTranscript?.trim() || "";
}

async function copyTranscript() {
  const text = transcriptText();
  if (!text) return toast("Nothing to copy yet");
  try {
    await navigator.clipboard.writeText(text);
    toast("Transcript copied");
  } catch {
    toast("Clipboard blocked by the browser");
  }
}

function downloadTranscript() {
  const text = transcriptText();
  if (!text) return toast("Nothing to download yet");

  const lines = [
    `Signal transcript — ${new Date().toLocaleString()}`,
    "",
    text,
    "",
    "Word log",
    ...state.entries.map((e) => `  ${e.time}  ${e.word}`),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `signal-transcript-${Date.now()}.txt`;
  link.click();
  URL.revokeObjectURL(url);
  toast("Transcript downloaded");
}

function openPresenter() {
  if (state.presenter && !state.presenter.closed) {
    state.presenter.focus();
    return;
  }
  state.presenter = window.open("./presenter.html", "signal-captions", "width=980,height=260");
  if (!state.presenter) return toast("Popup blocked — allow popups for this site");
  toast("Share that window in your call");
  // The new window subscribes on load, so resend once it is listening.
  setTimeout(render, 700);
}

function wire() {
  dom.startBtn.addEventListener("click", startCamera);

  document.querySelectorAll(".seg").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  dom.holdRange.addEventListener("input", (e) => {
    settings.holdMs = Number(e.target.value);
    dom.holdOut.textContent = `${(settings.holdMs / 1000).toFixed(2)}s`;
    saveSettings();
  });
  dom.pauseRange.addEventListener("input", (e) => {
    settings.pauseMs = Number(e.target.value);
    if (state.words) state.words.pauseMs = settings.pauseMs;
    dom.pauseOut.textContent = `${(settings.pauseMs / 1000).toFixed(2)}s`;
    saveSettings();
  });

  dom.speakToggle.addEventListener("change", (e) => {
    settings.speak = e.target.checked;
    state.speaker.enabled = settings.speak;
    dom.voiceFields.hidden = !settings.speak || !state.speaker.voices.length;
    if (!settings.speak) state.speaker.stop();
    saveSettings();
  });
  dom.voiceSelect.addEventListener("change", (e) => {
    settings.voiceURI = e.target.value;
    state.speaker.voiceURI = settings.voiceURI;
    saveSettings();
  });
  dom.rateRange.addEventListener("input", (e) => {
    settings.rate = Number(e.target.value);
    state.speaker.rate = settings.rate;
    dom.rateOut.textContent = `${settings.rate.toFixed(1)}×`;
    saveSettings();
  });
  dom.speakAllBtn.addEventListener("click", () => {
    const text = transcriptText();
    if (!text) return toast("Nothing to read yet");
    state.speaker.say(text);
  });

  dom.spaceBtn.addEventListener("click", () => {
    state.words?.addSpace();
    state.lastCommitted = "";
    render();
  });
  dom.backBtn.addEventListener("click", () => {
    state.words?.backspace();
    render();
  });
  dom.undoBtn.addEventListener("click", () => {
    if (state.words?.undo()) {
      state.lastCommitted = "";
      render();
    }
  });
  dom.clearBtn.addEventListener("click", () => {
    state.words?.clear();
    state.lastCommitted = "";
    state.entries = [];
    dom.history.replaceChildren(emptyHistoryRow());
    render();
    toast("Transcript cleared");
  });
  dom.copyBtn.addEventListener("click", copyTranscript);
  dom.downloadBtn.addEventListener("click", downloadTranscript);
  dom.presentBtn.addEventListener("click", openPresenter);

  dom.skeletonToggle.addEventListener("change", (e) => {
    settings.skeleton = e.target.checked;
    saveSettings();
  });
  dom.mirrorToggle.addEventListener("change", (e) => {
    settings.mirror = e.target.checked;
    dom.viewport.classList.toggle("mirrored", settings.mirror);
    saveSettings();
  });
  dom.cameraSelect.addEventListener("change", (e) => {
    settings.cameraId = e.target.value;
    saveSettings();
    startCamera();
  });

  dom.themeBtn.addEventListener("click", () => {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const current = settings.theme || (dark ? "dark" : "light");
    settings.theme = current === "dark" ? "light" : "dark";
    applyTheme();
    saveSettings();
  });

  dom.guideBtn.addEventListener("click", () => dom.guideDialog.showModal());
  dom.guideClose.addEventListener("click", () => dom.guideDialog.close());
  dom.guideSearch.addEventListener("input", (e) => renderGuide(e.target.value));

  document.addEventListener("keydown", onKeyDown);
  window.addEventListener("beforeunload", () => {
    state.stream?.getTracks().forEach((t) => t.stop());
    state.speaker.stop();
  });
}

function emptyHistoryRow() {
  const li = document.createElement("li");
  li.className = "history-empty";
  li.textContent = "Committed words will be listed here with timestamps.";
  return li;
}

function onKeyDown(event) {
  if (dom.guideDialog.open) return;
  // event.target is not always an Element (it can be the document), and
  // Document has no matches() — guard before asking.
  const target = event.target;
  if (target instanceof Element && target.matches("input, textarea, select")) return;
  if (!state.words) return;

  const key = event.key.toLowerCase();

  if ((event.metaKey || event.ctrlKey) && key === "z") {
    event.preventDefault();
    if (state.words.undo()) {
      state.lastCommitted = "";
      render();
    }
    return;
  }
  if (event.metaKey || event.ctrlKey || event.altKey) return;

  if (event.key === " ") {
    event.preventDefault();
    state.words.addSpace();
    state.lastCommitted = "";
  } else if (event.key === "Backspace") {
    event.preventDefault();
    state.words.backspace();
  } else if (key === "c") {
    state.words.clear();
    state.lastCommitted = "";
    state.entries = [];
    dom.history.replaceChildren(emptyHistoryRow());
  } else if (key === "v") {
    return void copyTranscript();
  } else if (key === "s") {
    return void state.speaker.say(transcriptText());
  } else if (key === "h") {
    dom.skeletonToggle.checked = !dom.skeletonToggle.checked;
    settings.skeleton = dom.skeletonToggle.checked;
    saveSettings();
    return;
  } else if (key === "m") {
    const next = MODE_ORDER[(MODE_ORDER.indexOf(settings.mode) + 1) % MODE_ORDER.length];
    setMode(next);
    toast(MODE_PILLS[next]);
    return;
  } else if (event.key === "?") {
    dom.guideDialog.showModal();
    return;
  } else {
    return;
  }
  render();
}

/* ----------------------------------------------------------------- init */

function init() {
  if (!navigator.mediaDevices?.getUserMedia) {
    dom.startBtn.disabled = true;
    dom.onboardNote.textContent =
      "This browser has no camera API. Try Chrome, Edge, or Safari over HTTPS.";
    setStatus(dom.statusCamera, "unsupported", "bad");
    return;
  }

  state.channel = new BroadcastChannel(CHANNEL);
  applySettingsToUI();
  wire();

  loadAssets().catch((error) => {
    setStatus(dom.statusModel, "failed", "bad");
    dom.onboardNote.textContent = `Could not load the recognizer: ${error.message}`;
  });
}

init();
