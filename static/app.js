import { Avatar, AudioQueue, PushToTalk } from "/static/avatar.js";

const $ = (id) => document.getElementById(id);

let models = [];
let conversations = [];
let active = null;       // {conv, ws}
let msgEls = {};         // msg_id -> element
let avatar = null;
let audioQueue = null;
let ptt = null;

// iOS Safari keeps AudioContext muted until resume() runs synchronously inside
// a user gesture. Prime it on the first pointerdown anywhere on the page.
window.__primedCtx = null;
function primeAudioOnce() {
  document.removeEventListener("click", primeAudioOnce, true);
  document.removeEventListener("touchend", primeAudioOnce, true);
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return;
  const ctx = new Ctx();
  window.__primedCtx = ctx;
  const src = ctx.createBufferSource();
  src.buffer = ctx.createBuffer(1, 1, 22050);
  src.connect(ctx.destination);
  try { src.start(0); } catch {}
  ctx.resume().catch(() => {});
}
document.addEventListener("click", primeAudioOnce, true);
document.addEventListener("touchend", primeAudioOnce, true);

function stopEverything() {
  // Always stop local audio/frame playback.
  if (audioQueue) audioQueue.stop();
  if (avatar) avatar.stopSpeaking();
  // Only send the server-side abort if a reply is actually in flight.
  // Otherwise the abort would close the bridge session (heavy ~5s reopen)
  // for no reason — e.g. on idle mic-press to start a new question.
  if (active && active.conv && active.conv.running &&
      active.ws && active.ws.readyState === WebSocket.OPEN) {
    active.ws.send(JSON.stringify({ type: "abort" }));
  }
}

// ---------- boot ----------
(async function init() {
  const r = await fetch("/api/models");
  const data = await r.json();
  models = data.models || [];
  if (data.error) console.warn("ollama:", data.error);
  await refreshList();
  wireModal();
  $("new-btn").onclick = openModal;
})();

// ---------- sidebar ----------
async function refreshList() {
  const r = await fetch("/api/conversations");
  conversations = await r.json();
  const ul = $("conv-list");
  ul.innerHTML = "";
  for (const c of conversations) {
    const li = document.createElement("li");
    if (active && active.conv.id === c.id) li.classList.add("active");
    li.innerHTML = `
      <div>
        <div>${escapeHtml(c.title)}</div>
        <div class="mode-badge">${c.mode.replace("_", " ↔ ")}${c.running ? " · running" : ""}</div>
      </div>
      <button class="del" title="delete">×</button>
    `;
    li.onclick = (e) => {
      if (e.target.classList.contains("del")) return;
      openConv(c.id);
    };
    li.querySelector(".del").onclick = async (e) => {
      e.stopPropagation();
      await fetch(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (active && active.conv.id === c.id) closeConv();
      refreshList();
    };
    ul.appendChild(li);
  }
}

// ---------- conversation view ----------
async function openConv(id) {
  closeConv();
  const r = await fetch(`/api/conversations/${id}`);
  const conv = await r.json();
  const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${wsProto}//${location.host}/ws/${id}`);
  ws.binaryType = "arraybuffer";
  active = { conv, ws };
  msgEls = {};

  $("empty").hidden = true;
  $("chat").hidden = false;
  renderHeader();
  $("messages").innerHTML = "";
  for (const m of conv.messages) renderMessage(m);

  ws.onmessage = (e) => {
    if (typeof e.data === "string") {
      handleEvent(JSON.parse(e.data));
    } else {
      // Binary frame: first byte is a type tag, rest is the payload.
      const view = new Uint8Array(e.data);
      const tag = view[0];
      const payload = e.data.slice(1);
      if (tag === 0x01 && avatar) avatar.paintJpeg(payload);
    }
  };
  ws.onclose = () => {};
  refreshList();
}

function closeConv() {
  if (active && active.ws) try { active.ws.close(); } catch {}
  active = null;
  $("empty").hidden = false;
  $("chat").hidden = true;
}

function renderHeader() {
  const { conv } = active;
  $("chat-title").textContent = conv.title;
  const parts = conv.participants.map((p) => `${p.name}${p.model ? ` (${p.model})` : ""}`).join(" ↔ ");
  const ref = "";
  const av = conv.avatar_mode ? ` · 🎤 avatar` : "";
  $("chat-meta").textContent = `${parts}${ref}${av} · turns ${conv.turns_taken}/${conv.max_turns}`;

  // Avatar pane is visible iff this conversation has avatar_mode on.
  const avatarOn = !!conv.avatar_mode;
  $("avatar-pane").hidden = !avatarOn;
  $("messages").style.flex = avatarOn ? "0 0 30%" : "1";
  if (avatarOn) ensureAvatar();

  const controls = $("chat-controls");
  controls.innerHTML = "";
  if (conv.mode === "llm_llm") {
    const start = btn(conv.running ? "⏸" : "▶", "primary", () => {
      if (conv.running) active.ws.send(JSON.stringify({ type: "stop" }));
      else active.ws.send(JSON.stringify({ type: "start" }));
    });
    controls.appendChild(start);
    if (conv.running) {
      controls.appendChild(btn("abort", "danger", () => active.ws.send(JSON.stringify({ type: "abort" }))));
    }
    controls.appendChild(btn("reset turns", "", () => active.ws.send(JSON.stringify({ type: "reset_turns" }))));
  }
  controls.appendChild(btn("⏹ stop", "danger", stopEverything));
  const thinkOn = !!conv.thinking_mode;
  controls.appendChild(btn(thinkOn ? "🧠 thinking: on" : "🧠 thinking: off", thinkOn ? "primary" : "", async () => {
    const r = await fetch(`/api/conversations/${conv.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thinking_mode: !thinkOn }),
    });
    const updated = await r.json();
    active.conv = updated;
    renderHeader();
  }));
  controls.appendChild(btn("edit", "", openEditModal));
  controls.appendChild(btn("clear", "", () => {
    active.ws.send(JSON.stringify({ type: "clear_messages" }));
  }));

  $("composer").hidden = !conv.participants.some((p) => p.kind === "user");
  $("composer").onsubmit = (e) => {
    e.preventDefault();
    const text = $("input").value.trim();
    if (!text) return;
    if (audioQueue) audioQueue.reset();
    if (avatar) avatar.reset();
    active.ws.send(JSON.stringify({ type: "user_message", content: text }));
    $("input").value = "";
  };
  $("input").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("composer").requestSubmit();
    }
  };
}

async function ensureAvatar() {
  if (audioQueue) return;
  if (!avatar) avatar = new Avatar($("avatar-canvas"));
  audioQueue = new AudioQueue(avatar);
  ptt = new PushToTalk({
    button: $("talk-btn"),
    onPressStart: stopEverything,
    onTranscript: (text) => {
      if (!active) return;
      if (audioQueue) audioQueue.reset();
      if (avatar) avatar.reset();
      active.ws.send(JSON.stringify({ type: "user_message", content: text }));
    },
  });
  ptt.bindElement($("avatar-canvas"));
  $("avatar-canvas").style.touchAction = "none";
  avatar.load().catch((e) => console.error("avatar load failed:", e));
}

function btn(label, cls, onClick) {
  const b = document.createElement("button");
  b.textContent = label;
  if (cls) b.className = cls;
  b.onclick = onClick;
  return b;
}

function handleEvent(ev) {
  switch (ev.type) {
    case "snapshot":
      active.conv = ev.conversation;
      renderHeader();
      $("messages").innerHTML = "";
      msgEls = {};
      ev.conversation.messages.forEach(renderMessage);
      break;
    case "message_start":
      renderMessage(ev.message);
      break;
    case "token": {
      const el = msgEls[ev.id];
      if (el) {
        el.querySelector(".bubble").textContent += ev.delta;
        scrollBottom();
      }
      break;
    }
    case "tts_chunk": {
      if (audioQueue) audioQueue.enqueueWavBase64(ev.audio_b64, ev.text || "");
      break;
    }
    case "avatar_ready": {
      if (avatar) avatar.onReady();
      break;
    }
    case "thinking": {
      const el = msgEls[ev.id];
      if (el) {
        el.querySelector(".thinking").textContent += ev.delta;
        if ($("messages").classList.contains("show-thinking")) scrollBottom();
      }
      break;
    }
    case "message_end":
      break;
    case "replace_content": {
      const el = msgEls[ev.id];
      if (el) {
        el.querySelector(".bubble").textContent = ev.content;
        scrollBottom();
      }
      break;
    }
    case "promote_thinking": {
      const el = msgEls[ev.id];
      if (el) el.querySelector(".thinking").textContent = "";
      break;
    }
    case "state":
      if (typeof ev.running === "boolean") active.conv.running = ev.running;
      if (typeof ev.turns_taken === "number") active.conv.turns_taken = ev.turns_taken;
      renderHeader();
      refreshList();
      break;
    case "error":
      console.error(ev.msg);
      break;
  }
}

function renderMessage(m) {
  const { conv } = active;
  const el = document.createElement("div");
  el.className = "msg";
  let who = "";
  if (m.sender < 0) {
    el.classList.add("system");
    who = "system";
  } else {
    const p = active.conv.participants[m.sender];
    who = p.name;
    if (p.kind === "user") { el.classList.add("user", "me"); }
    else { el.classList.add("llm", `llm-${m.sender}`); }
  }
  el.innerHTML = `<div class="who">${escapeHtml(who)}</div><div class="bubble"></div><div class="thinking"></div>`;
  el.querySelector(".bubble").textContent = m.content;
  if (m.thinking) el.querySelector(".thinking").textContent = m.thinking;
  $("messages").appendChild(el);
  msgEls[m.id] = el;
  scrollBottom();
}

function scrollBottom() {
  const m = $("messages");
  m.scrollTop = m.scrollHeight;
}

// ---------- modal ----------
function wireModal() {
  $("f-cancel").onclick = closeModal;
  $("f-create").onclick = createConv;
  $("edit-cancel").onclick = () => { $("edit-modal").hidden = true; };
  $("edit-save").onclick = saveEditModal;
  $("f-mode").onchange = renderParticipantFields;
  // Avatar toggle only affects the voice dropdown's visibility — DON'T
  // re-render participant cards (that wipes anything typed into the system
  // prompt textarea).
  $("f-avatar-mode").onchange = () => {
    const allowAvatar = $("f-mode").value === "user_llm";
    $("avatar-voice-label").hidden = !allowAvatar || !$("f-avatar-mode").checked;
  };
}

function openModal() {
  $("modal").hidden = false;
  renderParticipantFields();
}
function closeModal() { $("modal").hidden = true; }

// Preferred default when populating a model dropdown. If a matching model is
// installed, it gets pre-selected; otherwise we fall back to the first option.
const DEFAULT_MODEL = "qwen3.6:35b";

function fillModelSelect(sel) {
  sel.innerHTML = "";
  for (const m of models) {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    sel.appendChild(o);
  }
  if (models.includes(DEFAULT_MODEL)) sel.value = DEFAULT_MODEL;
}

function renderParticipantFields() {
  const mode = $("f-mode").value;
  const host = $("participants-fields");
  host.innerHTML = "";
  $("max-turns-label").hidden = mode !== "llm_llm";
  const allowAvatar = mode === "user_llm";
  $("avatar-mode-label").hidden = !allowAvatar;
  $("avatar-voice-label").hidden = !allowAvatar || !$("f-avatar-mode").checked;

  const mkLLM = (i, defaultName) => {
    const card = document.createElement("div");
    card.className = "participant-card";
    card.innerHTML = `
      <h3>participant ${i + 1} (llm)</h3>
      <label>name <input data-k="name" value="${defaultName}" /></label>
      <label>model <select data-k="model"></select></label>
      <label>system prompt <textarea data-k="system" rows="3"></textarea></label>
    `;
    fillModelSelect(card.querySelector("select"));
    host.appendChild(card);
    return card;
  };

  if (mode === "user_llm") {
    const userCard = document.createElement("div");
    userCard.className = "participant-card";
    userCard.innerHTML = `
      <h3>participant 1 (user)</h3>
      <label>name <input data-k="name" value="you" /></label>
    `;
    host.appendChild(userCard);
    mkLLM(1, "assistant");
  } else {
    mkLLM(0, "agent-a");
    mkLLM(1, "agent-b");
  }
}

async function createConv() {
  const mode = $("f-mode").value;
  const cards = document.querySelectorAll(".participant-card");
  const participants = [];
  cards.forEach((card, i) => {
    const p = { name: card.querySelector('[data-k=name]').value || `p${i}`, kind: "llm" };
    if (mode === "user_llm" && i === 0) p.kind = "user";
    if (p.kind === "llm") {
      p.model = card.querySelector('[data-k=model]').value;
      p.system = card.querySelector('[data-k=system]').value || null;
    }
    participants.push(p);
  });

  const body = {
    title: $("f-title").value || null,
    mode,
    participants,
    max_turns: parseInt($("f-max-turns").value || "20", 10),
    avatar_mode: mode === "user_llm" && $("f-avatar-mode").checked,
    tts_voice: $("f-tts-voice").value,
  };

  const r = await fetch("/api/conversations", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const conv = await r.json();
  closeModal();
  await refreshList();
  openConv(conv.id);
}

// ---------- edit modal ----------
function openEditModal() {
  if (!active) return;
  const { conv } = active;
  const body = $("edit-body");
  body.innerHTML = "";

  conv.participants.forEach((p, i) => {
    if (p.kind !== "llm") return;
    const card = document.createElement("div");
    card.className = "participant-card";
    card.dataset.index = i;
    card.innerHTML = `
      <h3>${escapeHtml(p.name)} (llm)</h3>
      <label>name <input data-k="name" value="${escapeHtml(p.name)}" /></label>
      <label>model <select data-k="model"></select></label>
      <label>system prompt <textarea data-k="system" rows="5"></textarea></label>
    `;
    const sel = card.querySelector("select");
    fillModelSelect(sel);
    sel.value = p.model || "";
    card.querySelector('[data-k=system]').value = p.system || "";
    body.appendChild(card);
  });

  if (conv.mode === "user_llm") {
    const card = document.createElement("div");
    card.className = "participant-card";
    card.dataset.av = "1";
    card.innerHTML = `
      <h3>avatar + voice</h3>
      <label><input data-k="avatar_mode" type="checkbox" /> avatar mode</label>
      <label>voice
        <select data-k="tts_voice">
          <option value="af_heart">af_heart (US female)</option>
          <option value="af_bella">af_bella (US female)</option>
          <option value="am_michael">am_michael (US male)</option>
          <option value="am_adam">am_adam (US male)</option>
          <option value="bf_emma">bf_emma (UK female)</option>
          <option value="bm_george">bm_george (UK male)</option>
        </select>
      </label>
    `;
    card.querySelector('[data-k=avatar_mode]').checked = !!conv.avatar_mode;
    card.querySelector('[data-k=tts_voice]').value = conv.tts_voice || "af_heart";
    body.appendChild(card);
  }

  $("edit-modal").hidden = false;
}

async function saveEditModal() {
  const { conv } = active;
  const cards = $("edit-body").querySelectorAll(".participant-card");
  const patch = { participants: [] };
  cards.forEach((card) => {
    if (card.dataset.av === "1") {
      patch.avatar_mode = card.querySelector('[data-k=avatar_mode]').checked;
      patch.tts_voice = card.querySelector('[data-k=tts_voice]').value;
      return;
    }
    {
      patch.participants.push({
        index: parseInt(card.dataset.index, 10),
        name: card.querySelector('[data-k=name]').value,
        model: card.querySelector('[data-k=model]').value,
        system: card.querySelector('[data-k=system]').value,
      });
    }
  });
  await fetch(`/api/conversations/${conv.id}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(patch),
  });
  $("edit-modal").hidden = true;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
