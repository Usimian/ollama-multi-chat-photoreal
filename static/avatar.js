// Photoreal avatar: paints JPEG frames from the server onto a canvas, and
// plays Kokoro TTS audio in step (lipsync is rendered server-side by Ditto,
// so the client just decodes images and plays audio).

export class Avatar {
  constructor(containerEl) {
    // Caller passes either a <canvas> or a wrapper element; make sure we end
    // up drawing into a canvas.
    let canvas = containerEl;
    if (canvas.tagName !== "CANVAS") {
      const c = document.createElement("canvas");
      c.id = containerEl.id || "avatar-canvas";
      c.className = containerEl.className || "";
      c.style.cssText = containerEl.style.cssText;
      containerEl.replaceWith(c);
      canvas = c;
    }
    canvas.width = 512;
    canvas.height = 512;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    // Backing audio context, lazily created. We don't actually need
    // sample-accurate timing here — the server-side frames are already locked
    // to the TTS audio — so a regular AudioContext is fine.
    this._audioCtx = null;
    this._frameCount = 0;
    this._ready = false;
    // Drop incoming frames after a stop until the next user-message reset.
    // Without this, server-side TTS-in-flight keeps painting after stop.
    this._stopped = false;

    // Frame buffer (decoded ImageBitmaps) + 25 fps painter. The worker runs
    // ~2.6× realtime, so frames arrive in bursts; buffering and playing at
    // 25 fps keeps the avatar locked to the audio's wall-clock rate.
    this._frameBuf = [];
    this._maxBuf = 250;          // ~10 s lookahead cap; drop oldest beyond.
    this._tickHandle = null;
    this._lastBitmap = null;

    // Audio playback is gated until the first frame is buffered, so audio and
    // video start aligned despite the ~1.6 s pipeline lead-in.
    this._firstFrameResolvers = [];

    // Idle: paint the still reference image immediately.
    this._paintMessage("loading avatar…");
    this._loadIdleStill();
  }

  // Awaitable: resolves on the next first frame OR after a hard timeout, so
  // short replies (under Ditto's ~5 s clip floor, which produce no frames at
  // all) still get their audio played instead of silently hanging.
  _waitForFirstFrame(timeoutMs = 4000) {
    return new Promise((resolve) => {
      let done = false;
      const finish = () => { if (!done) { done = true; resolve(); } };
      this._firstFrameResolvers.push(finish);
      setTimeout(finish, timeoutMs);
    });
  }

  _flushFirstFrameWaiters() {
    const w = this._firstFrameResolvers;
    this._firstFrameResolvers = [];
    for (const r of w) r();
  }

  // Simple 25 fps painter that pops one bitmap per tick. Imperfect sync (we
  // know about the drift) but at least it actually plays.
  _startTick() {
    if (this._tickHandle) return;
    this._tickHandle = setInterval(() => {
      const bm = this._frameBuf.shift();
      if (bm) {
        this._paintBitmap(bm);
        this._lastBitmap?.close?.();
        this._lastBitmap = bm;
      }
    }, 40);
  }

  async _loadIdleStill() {
    try {
      const r = await fetch("/assets/avatar_ref.jpg", { cache: "force-cache" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const blob = await r.blob();
      const bitmap = await createImageBitmap(blob);
      this._idleBitmap = bitmap;
      // Only paint the still if no live frame has arrived yet.
      if (this._frameCount === 0) this._paintBitmap(bitmap);
    } catch (e) {
      console.warn("idle still failed to load:", e);
    }
  }

  _paintBitmap(bitmap) {
    const { ctx, canvas } = this;
    const { width: cw, height: ch } = canvas;
    const s = Math.min(cw / bitmap.width, ch / bitmap.height);
    const dw = bitmap.width * s;
    const dh = bitmap.height * s;
    const dx = (cw - dw) / 2;
    const dy = (ch - dh) / 2;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cw, ch);
    ctx.drawImage(bitmap, dx, dy, dw, dh);
  }

  get audioCtx() {
    if (!this._audioCtx) this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    return this._audioCtx;
  }

  _paintMessage(text) {
    const { ctx, canvas } = this;
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#888";
    ctx.font = "16px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  }

  // Server signals the Ditto session is open and frames will start arriving.
  onReady() {
    this._ready = true;
    // If the idle still hasn't loaded yet, the loading splash stays. If it has,
    // we're already showing the reference portrait — leave it alone until live
    // frames begin.
  }

  // Enqueue a JPEG frame received over the websocket. The 25 fps tick consumes
  // it later. Both _startTick and _flushFirstFrameWaiters are idempotent, so
  // we run them on every frame and avoid any race around "is this the first
  // frame of a new utterance".
  async paintJpeg(arrayBuffer) {
    if (this._stopped) return;
    try {
      const blob = new Blob([arrayBuffer], { type: "image/jpeg" });
      const bitmap = await createImageBitmap(blob);
      if (this._stopped) { bitmap.close?.(); return; }
      this._frameBuf.push(bitmap);
      while (this._frameBuf.length > this._maxBuf) {
        const dropped = this._frameBuf.shift();
        dropped?.close?.();
      }
      this._frameCount += 1;
      if (this._frameCount === 1 || this._frameCount % 25 === 0) {
        console.log("[frame] paintJpeg ok, frameCount=", this._frameCount, "buf=", this._frameBuf.length, "waiters=", this._firstFrameResolvers.length);
      }
      this._startTick();
      this._flushFirstFrameWaiters();
    } catch (e) {
      console.error("paintJpeg failed:", e);
    }
  }

  // Compatibility shims — app.js still calls these.
  async load() { /* no-op; canvas is ready immediately */ }

  // Stop button: clear pending frames, halt the painter, paint the idle still,
  // and block further frames until reset() is called.
  stopSpeaking() {
    this._stopped = true;
    for (const bm of this._frameBuf) bm?.close?.();
    this._frameBuf = [];
    if (this._tickHandle) { clearInterval(this._tickHandle); this._tickHandle = null; }
    this._flushFirstFrameWaiters();   // unblock any waiting audio so it won't hang
    if (this._idleBitmap) this._paintBitmap(this._idleBitmap);
  }

  // Re-arm for the next utterance (called when user sends a new message).
  // Do a full stopSpeaking() to tear down the painter/anchor/buffer state so
  // the *next* first-frame triggers a fresh _startTick + _flushFirstFrameWaiters
  // cycle. Without this, the rAF loop from the previous utterance keeps
  // running, _tickHandle stays truthy, wasEmpty in paintJpeg is never true,
  // and the gated audio waiter never gets resolved.
  reset() {
    this.stopSpeaking();
    this._stopped = false;
  }
}


// ---------- Audio queue: plays Kokoro WAV chunks in order ----------

export class AudioQueue {
  constructor(avatar) {
    this.avatar = avatar;
    this._chain = Promise.resolve();
    this._needFirstFrame = true;
    this._gen = 0;
    // Next-scheduled audio context time. When a chunk is scheduled, we set
    // this to its end-time so the next chunk lines up with no gap.
    this._nextStart = 0;
    this._active = null;
  }

  enqueueWavBase64(b64, text = "") {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const myGen = this._gen;
    console.log("[audio] enqueue chunk, bytes=", bytes.length, "needFirstFrame=", this._needFirstFrame, "gen=", myGen);
    if (this._needFirstFrame) {
      this._needFirstFrame = false;
      this._chain = this._chain
        .then(() => this.avatar._waitForFirstFrame())
        .then(() => this._play(bytes, myGen, /*isFirst*/ true))
        .catch((e) => console.error("audio play failed:", e));
    } else {
      this._chain = this._chain
        .then(() => this._play(bytes, myGen, /*isFirst*/ false))
        .catch((e) => console.error("audio play failed:", e));
    }
    clearTimeout(this._rearm);
    this._rearm = setTimeout(() => { this._needFirstFrame = true; }, 600);
  }

  async _play(wavBytes, gen, isFirst) {
    console.log("[audio] _play start gen=", gen, "this._gen=", this._gen, "isFirst=", isFirst, "bytes=", wavBytes.length);
    if (gen !== this._gen) { console.log("[audio] gen mismatch, skipping"); return; }
    const ctx = this.avatar.audioCtx;
    console.log("[audio] ctx.state=", ctx.state, "currentTime=", ctx.currentTime);
    if (ctx.state === "suspended") {
      try { await ctx.resume(); console.log("[audio] resumed, state=", ctx.state); } catch (e) { console.warn("[audio] resume failed:", e); }
    }
    let buf;
    try {
      buf = await ctx.decodeAudioData(wavBytes.buffer.slice(0));
      console.log("[audio] decoded, duration=", buf.duration, "channels=", buf.numberOfChannels, "rate=", buf.sampleRate);
    } catch (e) {
      console.error("[audio] decode failed:", e);
      return;
    }
    if (gen !== this._gen) { console.log("[audio] gen mismatch post-decode, skipping"); return; }
    return new Promise((resolve) => {
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.onended = () => { console.log("[audio] onended"); if (this._active === src) this._active = null; resolve(); };
      this._active = src;
      try {
        src.start();
        console.log("[audio] src.start() ok");
      } catch (e) {
        console.error("[audio] src.start() threw:", e);
        resolve();
      }
    });
  }

  stop() {
    this._gen += 1;
    try { this._active?.stop(); } catch {}
    this._active = null;
    this._chain = Promise.resolve();
    this._needFirstFrame = true;
    clearTimeout(this._rearm);
  }

  reset() { this.stop(); }
}


// ---------- Push-to-talk recorder (unchanged from the upstream rig) ----------

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
