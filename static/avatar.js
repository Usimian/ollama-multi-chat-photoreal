// Avatar renderer + audio queue + push-to-talk.
// Wraps met4citizen/TalkingHead so the chat app gets idle motion, blinks,
// micro-saccades, head sway, and proper viseme-driven lipsync.

import { TalkingHead } from "/static/vendor/talkinghead/talkinghead.mjs";

const DEFAULT_AVATAR_URL = "/static/avatar.glb";

export class Avatar {
  constructor(containerEl) {
    // TalkingHead needs a container DIV (it creates its own canvas inside).
    // If the caller passed a <canvas>, swap it out.
    let el = containerEl;
    if (el.tagName === "CANVAS") {
      const div = document.createElement("div");
      div.id = el.id;
      div.className = el.className;
      div.style.cssText = el.style.cssText;
      el.parentNode.replaceChild(div, el);
      el = div;
    }
    this.container = el;
    this.head = new TalkingHead(el, {
      ttsEndpoint: "",            // we feed pre-rendered audio
      lipsyncModules: ["en"],
      cameraView: "upper",
    });
    this._loaded = false;
  }

  async load(url = DEFAULT_AVATAR_URL) {
    await this.head.showAvatar({
      url,
      body: "F",
      avatarMood: "neutral",
      lipsyncLang: "en",
    });
    this._loaded = true;
    this.head.start();
  }

  // Build approximate word/duration timings for a sentence given the audio length.
  // TalkingHead's lipsync module converts these to viseme animations.
  _buildSpeechItem(text, audioBuffer) {
    const cleaned = (text || "").replace(/\s+/g, " ").trim();
    const words = cleaned ? cleaned.split(" ") : [];
    const totalMs = audioBuffer.duration * 1000;
    const item = { audio: audioBuffer };
    if (words.length) {
      const totalChars = words.reduce((n, w) => n + Math.max(1, w.length), 0);
      const msPerChar = totalMs / totalChars;
      const wtimes = [];
      const wdurations = [];
      let t = 0;
      for (const w of words) {
        const d = Math.max(80, w.length * msPerChar);
        wtimes.push(t);
        wdurations.push(d);
        t += d;
      }
      item.words = words;
      item.wtimes = wtimes;
      item.wdurations = wdurations;
    }
    return item;
  }

  // Called by AudioQueue to play a Kokoro WAV chunk with lipsync.
  async speak(text, wavBytes) {
    if (!this._loaded) {
      // If avatar mesh isn't loaded yet, fall back to plain playback.
      await this._playRaw(wavBytes);
      return;
    }
    const buf = await this.head.audioCtx.decodeAudioData(wavBytes.buffer.slice(0));
    const item = this._buildSpeechItem(text, buf);
    this.head.speakAudio(item);
  }

  stopSpeaking() {
    try { this.head.stopSpeaking(); } catch (e) { console.warn("stopSpeaking:", e); }
  }

  async _playRaw(wavBytes) {
    const ctx = this.head.audioCtx;
    const buf = await ctx.decodeAudioData(wavBytes.buffer.slice(0));
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start();
  }
}


// ---------- Audio queue: feeds Kokoro chunks into the TalkingHead speech queue ----------

export class AudioQueue {
  constructor(avatar) {
    this.avatar = avatar;
  }

  enqueueWavBase64(b64, text = "") {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    // TalkingHead has its own internal speech queue; just hand it off.
    this.avatar.speak(text, bytes).catch((e) => console.error("speak failed:", e));
  }

  reset() {
    if (this.avatar && this.avatar.head) {
      try { this.avatar.head.stopSpeaking(); } catch {}
    }
  }
}


// ---------- Push-to-talk recorder (unchanged behavior) ----------

export class PushToTalk {
  constructor({ onTranscript, onPressStart, button, sttUrl = "/api/stt" }) {
    this.onTranscript = onTranscript;
    this.onPressStart = onPressStart;
    this.button = button;
    this.sttUrl = sttUrl;
    this.recorder = null;
    this.chunks = [];
    this.recording = false;

    const begin = (e) => { e.preventDefault(); this.start(); };
    const end = (e) => { e.preventDefault(); this.stop(); };

    button.addEventListener("mousedown", begin);
    button.addEventListener("mouseup", end);
    button.addEventListener("mouseleave", () => { if (this.recording) this.stop(); });
    button.addEventListener("touchstart", begin);
    button.addEventListener("touchend", end);

    window.addEventListener("keydown", (e) => {
      if (e.code === "Space" && !e.repeat && !this._isTyping(e.target)) {
        e.preventDefault();
        this.start();
      }
    });
    window.addEventListener("keyup", (e) => {
      if (e.code === "Space" && !this._isTyping(e.target)) {
        e.preventDefault();
        this.stop();
      }
    });
  }

  _isTyping(el) {
    return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable);
  }

  _status(s) {
    const el = document.getElementById("mic-status");
    if (el) el.textContent = s;
    console.log("[ptt]", s);
  }

  async start() {
    if (this.recording) return;
    if (this.onPressStart) { try { this.onPressStart(); } catch (e) { console.warn(e); } }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : (MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "");
      this.recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
      this.mime = this.recorder.mimeType || "audio/webm";
      this.chunks = [];
      this.recorder.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
      this.recorder.onstop = () => this._finish(stream);
      this.recorder.start();
      this.recording = true;
      this.button.classList.add("recording");
      this._status("recording…");
    } catch (e) {
      console.error("mic error", e);
      this._status(`mic error: ${e.message}`);
    }
  }

  stop() {
    if (!this.recording || !this.recorder) return;
    this.recording = false;
    this.button.classList.remove("recording");
    this.recorder.stop();
  }

  async _finish(stream) {
    stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(this.chunks, { type: this.mime || "audio/webm" });
    this._status(`uploading ${(blob.size/1024).toFixed(1)} KB…`);
    if (blob.size < 800) { this._status("too short — try again"); return; }
    const form = new FormData();
    const ext = (this.mime || "").includes("webm") ? "webm" : "ogg";
    form.append("file", blob, `speech.${ext}`);
    try {
      const r = await fetch(this.sttUrl, { method: "POST", body: form });
      const data = await r.json();
      if (data.error) { this._status(`stt error: ${data.error}`); return; }
      if (!data.text) { this._status("stt: (no speech detected)"); return; }
      this._status(`you: ${data.text}`);
      if (this.onTranscript) this.onTranscript(data.text);
    } catch (e) {
      console.error("stt error", e);
      this._status(`stt error: ${e.message}`);
    }
  }
}
