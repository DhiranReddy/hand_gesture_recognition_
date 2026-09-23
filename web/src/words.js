/**
 * Fingerspelling buffer: auto-space on pause, spell-correct on commit.
 *
 * Ported from src/word_processor.py with one deliberate change — the Python
 * version counts frames (PAUSE_FRAMES = 22), so its pause window is ~1.5s on a
 * 15fps laptop and ~0.35s on a 60fps machine. Here everything is milliseconds,
 * so behaviour is identical on every device.
 */

const PAUSE_MS = 730;
const MIN_WORD_LENGTH = 2;
const ALPHABET = "abcdefghijklmnopqrstuvwxyz";

/** Norvig-style corrector over a rank-ordered frequency list. */
export class SpellCorrector {
  constructor(words) {
    this.rank = new Map();
    words.forEach((word, index) => {
      if (!this.rank.has(word)) this.rank.set(word, index);
    });
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Could not load wordlist (${response.status}) from ${url}`);
    }
    return new SpellCorrector(await response.json());
  }

  has(word) {
    return this.rank.has(word);
  }

  #edits1(word) {
    const out = new Set();
    for (let i = 0; i <= word.length; i += 1) {
      const left = word.slice(0, i);
      const right = word.slice(i);
      if (right) out.add(left + right.slice(1));
      if (right.length > 1) out.add(left + right[1] + right[0] + right.slice(2));
      for (const c of ALPHABET) {
        if (right) out.add(left + c + right.slice(1));
        out.add(left + c + right);
      }
    }
    out.delete(word);
    return out;
  }

  #best(candidates) {
    let winner = null;
    let bestRank = Infinity;
    for (const candidate of candidates) {
      const rank = this.rank.get(candidate);
      if (rank !== undefined && rank < bestRank) {
        winner = candidate;
        bestRank = rank;
      }
    }
    return winner;
  }

  correct(word) {
    const lower = word.toLowerCase();
    if (lower.length < MIN_WORD_LENGTH || this.has(lower)) return lower;

    const one = this.#edits1(lower);
    const near = this.#best(one);
    if (near) return near;

    // Distance 2 — only worth the work for short buffers.
    if (lower.length <= 7) {
      const two = new Set();
      for (const candidate of one) {
        for (const deeper of this.#edits1(candidate)) {
          if (this.has(deeper)) two.add(deeper);
        }
      }
      const far = this.#best(two);
      if (far) return far;
    }
    return lower;
  }
}

export class WordProcessor {
  constructor(corrector, { pauseMs = PAUSE_MS } = {}) {
    this.corrector = corrector;
    this.pauseMs = pauseMs;
    this.transcript = "";
    this.buffer = "";
    this.preview = "";
    this.lastCorrection = "";
    this.idleSince = null;
  }

  get displayTranscript() {
    return this.buffer ? `${this.transcript}${this.buffer}` : this.transcript;
  }

  setPreview(text) {
    this.preview = text || "";
  }

  clearPreview() {
    this.preview = "";
  }

  /**
   * Called every frame in which no gesture is being held.
   * @param {number} now high-resolution timestamp in ms
   * @returns {boolean} true when the buffer was committed
   */
  onIdle(now) {
    if (!this.buffer) {
      this.idleSince = null;
      return false;
    }
    if (this.idleSince === null) {
      this.idleSince = now;
      return false;
    }
    if (now - this.idleSince >= this.pauseMs) {
      this.flush();
      return true;
    }
    return false;
  }

  addGesture(text, category) {
    this.idleSince = null;
    const isChar = category === "letter" || category === "number"
      || (text.length === 1 && /[a-z0-9]/i.test(text));

    if (isChar) {
      this.buffer += /[a-z]/i.test(text) ? text.toUpperCase() : text;
      return;
    }
    this.flush();
    this.#append(text);
  }

  flush() {
    if (!this.buffer) return;
    const raw = this.buffer;
    const corrected = this.corrector ? this.corrector.correct(raw) : raw.toLowerCase();
    this.lastCorrection = corrected.toLowerCase() !== raw.toLowerCase() ? corrected : "";
    this.#append(corrected);
    this.buffer = "";
    this.idleSince = null;
  }

  #append(word) {
    if (!word) return;
    if (this.transcript && !this.transcript.endsWith(" ")) this.transcript += " ";
    this.transcript += word;
  }

  addSpace() {
    this.flush();
    if (this.transcript && !this.transcript.endsWith(" ")) this.transcript += " ";
  }

  backspace() {
    if (this.buffer) this.buffer = this.buffer.slice(0, -1);
    else if (this.transcript) this.transcript = this.transcript.slice(0, -1);
  }

  clear() {
    this.transcript = "";
    this.buffer = "";
    this.preview = "";
    this.lastCorrection = "";
    this.idleSince = null;
  }
}
