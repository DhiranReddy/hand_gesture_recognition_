import test from "node:test";
import assert from "node:assert/strict";

function formatTimestampVTT(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const millis = Math.floor(ms % 1000);
  const pad = (n, z = 2) => String(n).padStart(z, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}.${pad(millis, 3)}`;
}

function formatTimestampSRT(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const millis = Math.floor(ms % 1000);
  const pad = (n, z = 2) => String(n).padStart(z, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)},${pad(millis, 3)}`;
}

function buildVTT(text, entries, startTime) {
  const lines = ["WEBVTT - Signal Gesture Transcript", ""];
  const t0 = startTime || (entries.length ? entries[0].timestampMs : Date.now());
  entries.forEach((entry, i) => {
    const startMs = Math.max(0, (entry.timestampMs || Date.now()) - t0);
    const endMs = startMs + 2000;
    lines.push(String(i + 1));
    lines.push(`${formatTimestampVTT(startMs)} --> ${formatTimestampVTT(endMs)}`);
    lines.push(entry.word);
    lines.push("");
  });
  return lines.join("\n");
}

function buildSRT(text, entries, startTime) {
  const lines = [];
  const t0 = startTime || (entries.length ? entries[0].timestampMs : Date.now());
  entries.forEach((entry, i) => {
    const startMs = Math.max(0, (entry.timestampMs || Date.now()) - t0);
    const endMs = startMs + 2000;
    lines.push(String(i + 1));
    lines.push(`${formatTimestampSRT(startMs)} --> ${formatTimestampSRT(endMs)}`);
    lines.push(entry.word);
    lines.push("");
  });
  return lines.join("\n");
}

test("formatTimestampVTT and formatTimestampSRT pad hours, minutes, seconds, milliseconds", () => {
  assert.equal(formatTimestampVTT(0), "00:00:00.000");
  assert.equal(formatTimestampVTT(1250), "00:00:01.250");
  assert.equal(formatTimestampVTT(65432), "00:01:05.432");
  assert.equal(formatTimestampVTT(3665432), "01:01:05.432");

  assert.equal(formatTimestampSRT(0), "00:00:00,000");
  assert.equal(formatTimestampSRT(1250), "00:00:01,250");
  assert.equal(formatTimestampSRT(65432), "00:01:05,432");
  assert.equal(formatTimestampSRT(3665432), "01:01:05,432");
});

test("buildVTT outputs compliant WebVTT cue list", () => {
  const startTime = 1000000;
  const entries = [
    { word: "Hello", time: "10:00:00", timestampMs: 1000000 },
    { word: "World", time: "10:00:03", timestampMs: 1003000 },
  ];
  const vtt = buildVTT("Hello World", entries, startTime);
  assert.match(vtt, /^WEBVTT - Signal Gesture Transcript/);
  assert.match(vtt, /00:00:00\.000 --> 00:00:02\.000\nHello/);
  assert.match(vtt, /00:00:03\.000 --> 00:00:05\.000\nWorld/);
});

test("buildSRT outputs compliant SubRip cue list", () => {
  const startTime = 1000000;
  const entries = [
    { word: "Hello", time: "10:00:00", timestampMs: 1000000 },
    { word: "World", time: "10:00:03", timestampMs: 1003000 },
  ];
  const srt = buildSRT("Hello World", entries, startTime);
  assert.match(srt, /^1\n00:00:00,000 --> 00:00:02,000\nHello/);
  assert.match(srt, /2\n00:00:03,000 --> 00:00:05,000\nWorld/);
});
