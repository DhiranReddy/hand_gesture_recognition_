/**
 * Speaking the captions aloud.
 *
 * The point of the app is being understood by people who are not reading a
 * screen, so committed words can be voiced through the Web Speech API. Voices
 * load asynchronously and some platforms report none at all, so every path here
 * degrades to silence rather than throwing.
 */

const synth = window.speechSynthesis;

export class Speaker {
  constructor() {
    this.enabled = false;
    this.voiceURI = null;
    this.rate = 1;
    this.voices = [];
    this.onVoicesChanged = null;
    this.#loadVoices();
  }

  static get supported() {
    return typeof window !== "undefined" && "speechSynthesis" in window;
  }

  #loadVoices() {
    if (!Speaker.supported) return;
    const read = () => {
      const all = synth.getVoices();
      if (!all.length) return;
      this.voices = all
        .filter((v) => v.lang.startsWith("en"))
        .sort((a, b) => Number(b.localService) - Number(a.localService));
      if (!this.voices.length) this.voices = all;
      this.onVoicesChanged?.(this.voices);
    };
    read();
    synth.addEventListener?.("voiceschanged", read);
  }

  get voice() {
    return this.voices.find((v) => v.voiceURI === this.voiceURI) || this.voices[0] || null;
  }

  /**
   * Queue a phrase. Short fingerspelled words arrive one at a time, so this is
   * additive rather than interrupting — cancelling on every word would clip
   * most of them before they were audible.
   */
  speak(text) {
    if (!this.enabled || !Speaker.supported || !text?.trim()) return;
    const utterance = new SpeechSynthesisUtterance(text.trim());
    const voice = this.voice;
    if (voice) {
      utterance.voice = voice;
      utterance.lang = voice.lang;
    }
    utterance.rate = this.rate;
    try {
      synth.speak(utterance);
    } catch {
      /* Autoplay policy or a busy engine — silence is an acceptable fallback. */
    }
  }

  /** Read the whole transcript back, replacing anything already queued. */
  say(text) {
    if (!Speaker.supported || !text?.trim()) return;
    this.stop();
    const wasEnabled = this.enabled;
    this.enabled = true;
    this.speak(text);
    this.enabled = wasEnabled;
  }

  stop() {
    if (Speaker.supported) synth.cancel();
  }
}
