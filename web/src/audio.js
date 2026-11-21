/**
 * Audio feedback for hands-free assistive communication.
 *
 * Provides subtle auditory cues using Web Audio API:
 *   - lock: gentle soft tick when hold threshold reaches completion
 *   - space: soft double-tick on word auto-commit
 *   - clear: descending tone
 *
 * Runs 100% on-device with zero audio assets or network downloads.
 */

class SoundEffects {
  constructor() {
    this.ctx = null;
    this.enabled = false;
    this.volume = 0.18;
  }

  #ensureContext() {
    if (!this.ctx && typeof window !== "undefined") {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  playLock() {
    if (!this.enabled) return;
    const ctx = this.#ensureContext();
    if (!ctx) return;

    try {
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      // Warm chime: 580 Hz quickly decaying
      osc.frequency.setValueAtTime(580, now);
      osc.frequency.exponentialRampToValueAtTime(880, now + 0.08);

      gain.gain.setValueAtTime(this.volume, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.1);
    } catch {
      // Audio context restricted by autoplay policy
    }
  }

  playCommit() {
    if (!this.enabled) return;
    const ctx = this.#ensureContext();
    if (!ctx) return;

    try {
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "triangle";
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.exponentialRampToValueAtTime(660, now + 0.07);

      gain.gain.setValueAtTime(this.volume * 1.2, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.12);
    } catch {
      // Degrade gracefully
    }
  }
}

export const sfx = new SoundEffects();
