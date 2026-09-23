/**
 * Signal — browser entry point.
 *
 * Pipeline: getUserMedia -> MediaPipe HandLandmarker (WASM) -> feature vector
 * -> exported MLP -> hold/commit state machine -> captions. Every stage runs in
 * the tab; no frame ever leaves the device.
 */

import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

import { GestureClassifier, labelToText, categoryFor } from "./classifier.js";
import { SpellCorrector, WordProcessor } from "./words.js";

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
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

const CAPTION_CHANNEL = "signal-captions";
const HOLD_CIRCUMFERENCE = 119.4;

const el = (id) => document.getElementById(id);

const dom = {
  videoWrap: el("video-wrap"),
  video: el("video"),
  overlay: el("overlay"),
  startBtn: el("start-btn"),
  idleNote: el("idle-note"),
  hud: el("hud"),
  fpsChip: el("fps-chip"),
  predChip: el("pred-chip"),
  holdIndicator: el("hold-indicator"),
  holdFill: el("hold-fill"),
  holdLabel: el("hold-label"),
  captionText: el("caption-text"),
  captionPreview: el("caption-preview"),
  correction: el("correction"),
  spaceBtn: el("space-btn"),
  backBtn: el("back-btn"),
  copyBtn: el("copy-btn"),
  clearBtn: el("clear-btn"),
  skeletonToggle: el("skeleton-toggle"),
  mirrorToggle: el("mirror-toggle"),
  holdRange: el("hold-range"),
  holdOut: el("hold-out"),
  pauseRange: el("pause-range"),
  pauseOut: el("pause-out"),
  presentBtn: el("present-btn"),
  statusModel: el("status-model"),
  statusLabels: el("status-labels"),
  statusCamera: el("status-camera"),
  helpBtn: el("help-btn"),
  helpDialog: el("help-dialog"),
  helpClose: el("help-close"),
  labelGroups: el("label-groups"),
  toast: el("toast"),
};

const state = {
  landmarker: null,
  classifier: null,
  words: null,
  stream: null,
  running: false,
  holdMs: 330,
  cooldownMs: 600,
  showSkeleton: true,
  currentLabel: null,
  holdStart: 0,
  cooldownUntil: 0,
  lastCommitted: "",
  lastVideoTime: -1,
  frameTimes: [],
  channel: null,
};

/* ---------------------------------------------------------------- helpers */

let toastTimer = null;
function toast(message) {
  dom.toast.textContent = message;
  dom.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    dom.toast.hidden = true;
  }, 2200);
}

function setStatus(node, text, kind = "") {
  node.textContent = text;
  node.className = kind;
}

/* ------------------------------------------------------------------ setup */

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
  state.words = new WordProcessor(corrector, {
    pauseMs: Number(dom.pauseRange.value),
  });

  setStatus(dom.statusModel, "ready", "ok");
  setStatus(dom.statusLabels, String(classifier.labels.length));
  buildLabelGuide(classifier.labels);
}

function buildLabelGuide(labels) {
  const groups = {
    Letters: labels.filter((l) => /^[A-Z]$/.test(l)),
    Numbers: labels.filter((l) => /^[0-9]$/.test(l)),
    "Words & phrases": labels.filter((l) => l.length > 1),
  };

  dom.labelGroups.replaceChildren();
  for (const [title, items] of Object.entries(groups)) {
    if (!items.length) continue;
    const heading = document.createElement("h3");
    heading.textContent = `${title} (${items.length})`;
    const row = document.createElement("div");
    row.className = "chip-row";
    for (const item of items) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = labelToText(item);
      row.append(chip);
    }
    dom.labelGroups.append(heading, row);
  }
}

async function startCamera() {
  dom.startBtn.disabled = true;
  dom.startBtn.textContent = "Starting…";

  try {
    if (!state.landmarker) await loadAssets();

    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" },
      audio: false,
    });
    dom.video.srcObject = state.stream;
    await dom.video.play();

    dom.overlay.width = dom.video.videoWidth;
    dom.overlay.height = dom.video.videoHeight;
    dom.videoWrap.classList.add("live");
    dom.hud.hidden = false;
    setStatus(dom.statusCamera, "live", "ok");

    state.running = true;
    requestAnimationFrame(loop);
  } catch (error) {
    dom.startBtn.disabled = false;
    dom.startBtn.textContent = "Enable camera";
    const message = error?.name === "NotAllowedError"
      ? "Camera permission was denied. Allow it in your browser's site settings and try again."
      : `Could not start: ${error.message}`;
    dom.idleNote.textContent = message;
    setStatus(dom.statusCamera, "error", "error");
    setStatus(dom.statusModel, state.classifier ? "ready" : "failed",
      state.classifier ? "ok" : "error");
  }
}

/* ------------------------------------------------------------- draw + loop */

function drawSkeleton(ctx, landmarks, width, height) {
  ctx.lineWidth = Math.max(2, width / 320);
  ctx.strokeStyle = "rgba(94, 234, 212, 0.85)";
  ctx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
    ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
  const r = Math.max(2.5, width / 260);
  for (const point of landmarks) {
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updateFps(now) {
  state.frameTimes.push(now);
  while (state.frameTimes.length && now - state.frameTimes[0] > 1000) {
    state.frameTimes.shift();
  }
  dom.fpsChip.textContent = `${state.frameTimes.length} fps`;
}

function loop() {
  if (!state.running) return;
  const now = performance.now();
  const { video, overlay } = dom;

  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    updateFps(now);

    const result = state.landmarker.detectForVideo(video, now);
    const landmarks = result.landmarks?.[0] || null;
    const handedness = result.handednesses?.[0]?.[0]?.categoryName || "Right";

    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (landmarks && state.showSkeleton) {
      drawSkeleton(ctx, landmarks, overlay.width, overlay.height);
    }

    step(landmarks, handedness, now);
  }

  requestAnimationFrame(loop);
}

/**
 * Hold/commit state machine. A gesture must stay stable for `holdMs` before it
 * is written, then the same gesture is suppressed for `cooldownMs` so a held
 * hand does not repeat.
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
    if (words.onIdle(now)) state.lastCommitted = "";
    dom.predChip.textContent = landmarks ? "Reading…" : "No hand detected";
    dom.predChip.classList.remove("active");
    dom.holdIndicator.hidden = true;
    render();
    return;
  }

  dom.predChip.textContent = `${text} · ${Math.round(confidence * 100)}%`;
  dom.predChip.classList.add("active");
  words.setPreview(text);

  if (label !== state.currentLabel) {
    state.currentLabel = label;
    state.holdStart = now;
  }

  const held = now - state.holdStart;
  const progress = Math.min(1, held / state.holdMs);

  dom.holdIndicator.hidden = false;
  dom.holdLabel.textContent = text;
  dom.holdFill.style.strokeDashoffset = String(
    HOLD_CIRCUMFERENCE * (1 - progress),
  );

  if (progress >= 1 && now >= state.cooldownUntil && text !== state.lastCommitted) {
    words.addGesture(text, categoryFor(label));
    state.lastCommitted = text;
    state.cooldownUntil = now + state.cooldownMs;
    state.holdStart = now;
    words.clearPreview();
    dom.holdIndicator.hidden = true;
  }

  render();
}

function render() {
  const { words } = state;
  const committed = words.transcript;
  const buffer = words.buffer;

  dom.captionText.replaceChildren();
  if (!committed && !buffer) {
    const placeholder = document.createElement("span");
    placeholder.className = "caption-placeholder";
    placeholder.textContent = "Your signed words appear here…";
    dom.captionText.append(placeholder);
  } else {
    if (committed) dom.captionText.append(document.createTextNode(committed));
    if (buffer) {
      if (committed && !committed.endsWith(" ")) {
        dom.captionText.append(document.createTextNode(" "));
      }
      const span = document.createElement("span");
      span.className = "caption-buffer";
      span.textContent = buffer;
      dom.captionText.append(span);
    }
  }

  dom.captionPreview.textContent = words.preview ? `Holding: ${words.preview}` : "";
  dom.correction.textContent = words.lastCorrection
    ? `auto-corrected → ${words.lastCorrection}`
    : "";

  state.channel?.postMessage({ text: words.displayTranscript });
}

/* --------------------------------------------------------------- controls */

function wireControls() {
  dom.startBtn.addEventListener("click", startCamera);

  dom.spaceBtn.addEventListener("click", () => {
    state.words?.addSpace();
    state.lastCommitted = "";
    render();
  });
  dom.backBtn.addEventListener("click", () => {
    state.words?.backspace();
    render();
  });
  dom.clearBtn.addEventListener("click", () => {
    state.words?.clear();
    state.lastCommitted = "";
    render();
  });
  dom.copyBtn.addEventListener("click", copyTranscript);

  dom.skeletonToggle.addEventListener("change", (e) => {
    state.showSkeleton = e.target.checked;
  });
  dom.mirrorToggle.addEventListener("change", (e) => {
    dom.videoWrap.classList.toggle("mirrored", e.target.checked);
  });
  dom.videoWrap.classList.toggle("mirrored", dom.mirrorToggle.checked);

  dom.holdRange.addEventListener("input", (e) => {
    state.holdMs = Number(e.target.value);
    dom.holdOut.textContent = `${(state.holdMs / 1000).toFixed(2)}s`;
  });
  dom.pauseRange.addEventListener("input", (e) => {
    const value = Number(e.target.value);
    if (state.words) state.words.pauseMs = value;
    dom.pauseOut.textContent = `${(value / 1000).toFixed(2)}s`;
  });

  dom.presentBtn.addEventListener("click", openPresenterWindow);

  dom.helpBtn.addEventListener("click", () => dom.helpDialog.showModal());
  dom.helpClose.addEventListener("click", () => dom.helpDialog.close());

  document.addEventListener("keydown", onKeyDown);
}

async function copyTranscript() {
  const text = state.words?.displayTranscript?.trim();
  if (!text) return toast("Nothing to copy yet");
  try {
    await navigator.clipboard.writeText(text);
    toast("Transcript copied");
  } catch {
    toast("Clipboard blocked by browser");
  }
}

function onKeyDown(event) {
  if (event.target.matches("input, textarea") || dom.helpDialog.open) {
    if (event.key === "Escape") dom.helpDialog.close();
    return;
  }
  if (!state.words) return;

  const key = event.key.toLowerCase();
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
  } else if (key === "v") {
    copyTranscript();
    return;
  } else if (key === "h") {
    dom.skeletonToggle.checked = !dom.skeletonToggle.checked;
    state.showSkeleton = dom.skeletonToggle.checked;
    return;
  } else if (event.key === "?") {
    dom.helpDialog.showModal();
    return;
  } else {
    return;
  }
  render();
}

/** Popup window showing only captions — the thing you screen-share. */
function openPresenterWindow() {
  const popup = window.open("", "signal-captions", "width=900,height=220");
  if (!popup) return toast("Popup blocked — allow popups for this site");

  popup.document.write(`<!doctype html><html><head><meta charset="utf-8">
<title>Signal — Captions</title><style>
  html,body{margin:0;height:100%;background:#000;color:#fff;
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
  body{display:flex;align-items:center;padding:0 5vw;}
  p{margin:0;font-size:clamp(28px,7vw,64px);font-weight:700;line-height:1.2;
    letter-spacing:-0.02em;overflow-wrap:anywhere;}
  .empty{opacity:.35;font-weight:500;font-size:clamp(20px,4vw,34px);}
</style></head><body><p id="t" class="empty">Waiting for captions…</p>
<script>
  new BroadcastChannel(${JSON.stringify(CAPTION_CHANNEL)}).onmessage = (e) => {
    const node = document.getElementById("t");
    const text = (e.data && e.data.text || "").trim();
    node.textContent = text || "Waiting for captions…";
    node.className = text ? "" : "empty";
  };
<\/script></body></html>`);
  popup.document.close();
  toast("Share this window in your call");
}

/* ------------------------------------------------------------------- init */

function init() {
  if (!navigator.mediaDevices?.getUserMedia) {
    dom.startBtn.disabled = true;
    dom.idleNote.textContent =
      "This browser does not expose a camera API. Try Chrome, Edge, or Safari over HTTPS.";
    return;
  }
  state.channel = new BroadcastChannel(CAPTION_CHANNEL);
  wireControls();
  loadAssets().catch((error) => {
    setStatus(dom.statusModel, "failed", "error");
    dom.idleNote.textContent = `Could not load the recognizer: ${error.message}`;
  });
}

init();
