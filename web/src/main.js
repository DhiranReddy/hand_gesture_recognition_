/**
 * Signal — browser entry point.
 *
 * Pipeline: getUserMedia -> MediaPipe HandLandmarker (WASM) -> feature vector
 * -> exported MLP -> hold/commit state machine -> captions, speech, and the
 * presenter window. Every stage runs in the tab; no frame leaves the device.
 */


import {
  GestureClassifier,
  labelToText,
  categoryFor,
  MODE_ALL,
  MODE_WORDS,
  MODE_LETTERS,
} from "./classifier.js?v=13";
import { SpellCorrector, WordProcessor } from "./words.js?v=13";
import { Speaker } from "./speech.js?v=13";
import {
  groupLabels, shapeFor, MOTION_LETTERS, CONFUSABLE, VOCABULARY, PHRASE_REASONS,
} from "./guide.js?v=13";
import { renderHand, hasDiagram, COLLISIONS } from "./handshape.js?v=13";
import { Calibration, MIN_SAMPLES } from "./calibration.js?v=13";
import { Recognizer } from "./recognizer.js?v=13";

const TASKS_VERSION = "0.10.18";
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;

/**
 * Prefer the vendored runtime when tools/vendor_runtime.py has been run, so a
 * cold load does not depend on two third-party hosts. Falls back to the CDN so
 * a fresh clone still works without the vendoring step.
 */
async function resolveRuntime() {
  try {
    const probe = await fetch("./vendor/manifest.json", { method: "GET" });
    if (probe.ok) {
      const module = await import("../vendor/tasks-vision/vision_bundle.mjs");
      return {
        module,
        wasmBase: "./vendor/tasks-vision/wasm",
        task: "./vendor/hand_landmarker.task",
        source: "vendored",
      };
    }
  } catch {
    /* Not vendored, or unreadable — fall through to the CDN. */
  }
  return {
    module: await import(`${CDN}/vision_bundle.mjs`),
    wasmBase: `${CDN}/wasm`,
    task:
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/" +
      "hand_landmarker/float16/1/hand_landmarker.task",
    source: "cdn",
  };
}

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/** Bump when the exported model changes, so browsers refetch it. */
const ASSET_VERSION = "2";
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
    "mode-hint", "model-toggle", "hold-range", "hold-out", "pause-range", "pause-out",
    "speak-toggle", "voice-fields", "voice-select", "rate-range", "rate-out",
    "voice-hint",
    "space-btn", "undo-btn", "back-btn", "copy-btn", "clear-btn", "present-btn",
    "camera-field", "camera-select", "skeleton-toggle", "mirror-toggle",
    "status-model", "status-labels", "status-camera",
    "guide-btn", "guide-dialog", "guide-close", "guide-search", "guide-groups",
    "calib-select", "calib-record", "calib-forget", "calib-status", "calib-meter",
    "calib-progress", "calib-hint", "calib-summary",
    "calib-export", "calib-import", "calib-file",
    "toast",
  ].map((id) => [id.replace(/-([a-z])/g, (_, c) => c.toUpperCase()), el(id)]),
);

/* ------------------------------------------------------------- settings */

const defaults = {
  mode: MODE_ALL,
  holdMs: 330,
  pauseMs: 730,
  speak: false,
  voiceURI: null,
  rate: 1,
  skeleton: true,
  mirror: true,
  cameraId: null,
  useModel: true,
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
  recognizer: null,
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
  calibration: new Calibration(),
  recording: null,
};

const RECORD_TARGET = 15;
/** Space samples out so a burst of near-identical frames is not recorded. */
const RECORD_INTERVAL_MS = 120;

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

/* ----------------------------------------------------------- load model */

async function loadAssets() {
  setStatus(dom.statusModel, "loading…");
  const runtime = await resolveRuntime();
  const { FilesetResolver, HandLandmarker } = runtime.module;

  const [vision, classifier, corrector] = await Promise.all([
    FilesetResolver.forVisionTasks(runtime.wasmBase),
    GestureClassifier.load(`./public/model/gesture_mlp.json?v=${ASSET_VERSION}`),
    SpellCorrector.load(`./public/model/words.json?v=${ASSET_VERSION}`),
  ]);
  state.runtimeSource = runtime.source;

  state.landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: runtime.task, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.7,
    minTrackingConfidence: 0.6,
  });

  state.classifier = classifier;
  state.recognizer = new Recognizer({ classifier, calibration: state.calibration });
  state.recognizer.useModel = settings.useModel;
  state.recognizer.setMode(settings.mode);
  state.words = new WordProcessor(corrector, { pauseMs: settings.pauseMs });

  setStatus(
    dom.statusModel,
    runtime.source === "vendored" ? "ready (vendored)" : "ready (cdn)",
    "ok",
  );
  setStatus(dom.statusLabels, `${VOCABULARY.length} gestures`);
  buildGuide(VOCABULARY);
  buildCalibrationPicker(VOCABULARY);
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
      reason: PHRASE_REASONS[label] || null,
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

/**
 * One guide entry: a button that expands to show how the shape is formed.
 * The diagram is built on first open so the dialog does not construct 36 SVGs
 * before anyone asks for one.
 */
function guideRow(item) {
  const row = document.createElement("div");
  row.className = "guide-item";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "guide-trigger";
  button.setAttribute("aria-expanded", "false");

  const key = document.createElement("span");
  key.className = "guide-key";
  key.textContent = item.label.length === 1 ? item.label : item.text;

  const desc = document.createElement("span");
  desc.className = "guide-desc";
  desc.textContent = item.shape || item.text;
  if (item.motion) {
    desc.append(flagEl("motion", "Needs movement — held still it reads as another sign."));
  }
  if (item.confusable) {
    desc.append(flagEl("similar", "Differs from a nearby gesture by one finger."));
  }

  const chevron = document.createElement("span");
  chevron.className = "guide-chevron";
  chevron.setAttribute("aria-hidden", "true");
  chevron.textContent = "›";

  button.append(key, desc, chevron);

  const panel = document.createElement("div");
  panel.className = "guide-detail";
  panel.hidden = true;

  let built = false;
  button.addEventListener("click", () => {
    const open = !panel.hidden;
    // Accordion: only one entry expanded at a time keeps the dialog scannable.
    for (const other of dom.guideGroups.querySelectorAll(".guide-detail")) {
      other.hidden = true;
    }
    for (const other of dom.guideGroups.querySelectorAll(".guide-trigger")) {
      other.setAttribute("aria-expanded", "false");
      other.classList.remove("open");
    }
    if (open) return;

    if (!built) {
      buildGuideDetail(panel, item);
      built = true;
    }
    panel.hidden = false;
    button.setAttribute("aria-expanded", "true");
    button.classList.add("open");
    panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });

  row.append(button, panel);
  return row;
}

function buildGuideDetail(panel, item) {
  const diagram = document.createElement("div");
  diagram.className = "hand-figure";
  const svg = hasDiagram(item.label) ? renderHand(item.label) : null;
  if (svg) {
    diagram.append(svg);
  } else {
    const fallback = document.createElement("span");
    fallback.className = "hand-missing";
    fallback.textContent = "No diagram for this phrase yet";
    diagram.append(fallback);
  }

  const body = document.createElement("div");
  body.className = "hand-body";

  const how = document.createElement("p");
  how.className = "hand-how";
  how.textContent = item.shape || item.text;
  body.append(how);

  if (item.reason) {
    const why = document.createElement("p");
    why.className = "hand-why";
    why.textContent = item.reason;
    body.append(why);
  }

  const steps = document.createElement("ol");
  steps.className = "hand-steps";
  for (const line of [
    `Face your palm to the camera and form the shape above.`,
    `Hold steady until the ring around the gesture closes.`,
    item.label.length === 1
      ? `It joins the current word; pause to commit and auto-correct it.`
      : `It is written as a whole phrase straight away.`,
  ]) {
    const li = document.createElement("li");
    li.textContent = line;
    steps.append(li);
  }
  body.append(steps);

  if (item.motion) {
    body.append(
      calloutEl(
        "Motion sign",
        `${item.text} is drawn in the air in real ASL. This recogniser only sees
         still shapes, so held still it will read as another sign.`,
      ),
    );
  }

  const clash = COLLISIONS[item.label];
  if (clash) {
    body.append(
      calloutEl(
        `Looks like ${clash.with.join(", ")}`,
        `${clash.why} Switch the recognition mode to Letters or Words to rule out
         the other category.`,
      ),
    );
  }

  panel.append(diagram, body);
}

function calloutEl(title, text) {
  const box = document.createElement("div");
  box.className = "hand-callout";
  const strong = document.createElement("strong");
  strong.textContent = title;
  const span = document.createElement("span");
  span.textContent = text.replace(/\s+/g, " ").trim();
  box.append(strong, span);
  return box;
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
      list.append(guideRow(item));
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
    refreshCalibrationUI();

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
  const { recognizer, words } = state;
  const { label, confidence, category, source } = recognizer.recognize(
    landmarks,
    handedness,
    dom.overlay.width,
    dom.overlay.height,
  );

  if (landmarks && state.recording) {
    // Store pixel-space landmarks so a recording session is also a dataset.
    const pixels = landmarks.map((p) => ({
      x: p.x * dom.overlay.width,
      y: p.y * dom.overlay.height,
      z: p.z ?? 0,
    }));
    captureSample(pixels, handedness, now);
  }

  // Recording should not also be typing.
  if (state.recording) {
    dom.readoutLabel.textContent = `Recording ${labelToText(state.recording.label)}`;
    dom.readoutPct.textContent = `${state.recording.taken} / ${RECORD_TARGET}`;
    dom.readoutMeter.style.width = `${(state.recording.taken / RECORD_TARGET) * 100}%`;
    dom.hold.hidden = true;
    return;
  }

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

  dom.readoutLabel.textContent =
    source === "you" ? `${text} \u00b7 yours` : text;
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
    words.addGesture(text, category);
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

/* ---------------------------------------------------------- calibration */

function buildCalibrationPicker(labels) {
  dom.calibSelect.replaceChildren();
  for (const label of labels) {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = labelToText(label);
    dom.calibSelect.append(option);
  }
  refreshCalibrationUI();
}

function refreshCalibrationUI() {
  const { calibration } = state;
  const label = dom.calibSelect.value;
  const count = label ? calibration.countFor(label) : 0;
  const trained = calibration.trainedLabels.length;

  dom.calibSummary.textContent = trained
    ? `${trained} gesture${trained === 1 ? "" : "s"} tuned`
    : "not calibrated";

  dom.calibForget.disabled = count === 0 || Boolean(state.recording);
  dom.calibExport.disabled = calibration.total === 0;
  dom.calibRecord.disabled = !state.running || Boolean(state.recording);

  for (const option of dom.calibSelect.options) {
    const n = calibration.countFor(option.value);
    const base = labelToText(option.value);
    option.textContent = n ? `${base} — ${n} sample${n === 1 ? "" : "s"}` : base;
  }
  dom.calibSelect.value = label;

  if (state.recording) return;

  if (!state.running) {
    dom.calibHint.textContent = "Enable the camera to record samples.";
  } else if (!count) {
    dom.calibHint.textContent =
      `No samples yet. Hold the shape and record ${RECORD_TARGET} of them.`;
  } else if (count < MIN_SAMPLES) {
    dom.calibHint.textContent =
      `${count} recorded — ${MIN_SAMPLES - count} more before it is used.`;
  } else {
    dom.calibHint.textContent = `${count} samples — your version is matched first.`;
  }
}

function startRecording() {
  if (!state.running || state.recording) return;
  const label = dom.calibSelect.value;
  if (!label) return;

  state.recording = { label, taken: 0, lastAt: 0, failed: false };
  dom.calibStatus.hidden = false;
  dom.calibRecord.disabled = true;
  dom.calibForget.disabled = true;
  dom.calibHint.textContent =
    `Hold ${labelToText(label)} steady, moving your hand slightly between samples.`;
  updateRecordingMeter();
}

function captureSample(pixels, handedness, now) {
  const rec = state.recording;
  if (!rec || now - rec.lastAt < RECORD_INTERVAL_MS) return;

  rec.lastAt = now;
  if (!state.calibration.add(rec.label, pixels, handedness)) rec.failed = true;
  rec.taken += 1;
  updateRecordingMeter();

  if (rec.taken >= RECORD_TARGET) stopRecording();
}

function updateRecordingMeter() {
  const rec = state.recording;
  if (!rec) return;
  dom.calibProgress.textContent = `${rec.taken} / ${RECORD_TARGET}`;
  dom.calibMeter.style.width = `${(rec.taken / RECORD_TARGET) * 100}%`;
}

function stopRecording() {
  const rec = state.recording;
  state.recording = null;
  dom.calibStatus.hidden = true;
  state.recognizer?.reset();
  refreshCalibrationUI();

  if (!rec) return;
  toast(
    rec.failed
      ? "Storage full — some samples were not saved"
      : `Recorded ${rec.taken} samples for ${labelToText(rec.label)}`,
  );
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
  state.recognizer?.setMode(mode);
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
  setMode(settings.mode);

  dom.holdRange.value = settings.holdMs;
  dom.holdOut.textContent = `${(settings.holdMs / 1000).toFixed(2)}s`;
  dom.pauseRange.value = settings.pauseMs;
  dom.pauseOut.textContent = `${(settings.pauseMs / 1000).toFixed(2)}s`;

  dom.skeletonToggle.checked = settings.skeleton;
  dom.mirrorToggle.checked = settings.mirror;
  dom.viewport.classList.toggle("mirrored", settings.mirror);

  dom.modelToggle.checked = settings.useModel;
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

  dom.modelToggle.addEventListener("change", (e) => {
    settings.useModel = e.target.checked;
    if (state.recognizer) {
      state.recognizer.useModel = settings.useModel;
      state.recognizer.reset();
    }
    saveSettings();
    toast(settings.useModel ? "Using trained model for letters" : "Using shape rules only");
  });

  dom.calibExport.addEventListener("click", () => {
    const text = state.calibration.toJSONL();
    if (!text) return toast("No samples recorded yet");
    const blob = new Blob([text], { type: "application/jsonl" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "gesture_landmarks.jsonl";
    link.click();
    URL.revokeObjectURL(url);
    toast(`Exported ${state.calibration.total} samples`);
  });

  dom.calibImport.addEventListener("click", () => dom.calibFile.click());
  dom.calibFile.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const added = state.calibration.fromJSONL(await file.text());
      state.recognizer?.reset();
      refreshCalibrationUI();
      toast(added ? `Imported ${added} samples` : "No usable samples in that file");
    } catch (error) {
      toast(`Could not read that file: ${error.message}`);
    }
    event.target.value = "";
  });

  dom.calibSelect.addEventListener("change", refreshCalibrationUI);
  dom.calibRecord.addEventListener("click", startRecording);
  dom.calibForget.addEventListener("click", () => {
    const label = dom.calibSelect.value;
    state.calibration.clear(label);
    state.recognizer?.reset();
    refreshCalibrationUI();
    toast(`Forgot your samples for ${labelToText(label)}`);
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
