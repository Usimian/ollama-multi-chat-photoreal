import { Avatar, AudioQueue, PushToTalk } from "/static/avatar.js";

const $ = (id) => document.getElementById(id);

let models = [];
let conversations = [];
let active = null;       // {conv, ws}
let msgEls = {};         // msg_id -> element
let avatar = null;
let audioQueue = null;
let ptt = null;

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
  const ws = new WebSocket(`ws://${location.host}/ws/${id}`);
  active = { conv, ws };
  msgEls = {};

  $("empty").hidden = true;
  $("chat").hidden = false;
  renderHeader();
  $("messages").innerHTML = "";
  for (const m of conv.messages) renderMessage(m);

  ws.onmessage = (e) => handleEvent(JSON.parse(e.data));
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
  const ref = conv.referee ? ` · referee: ${conv.referee.model}` : "";
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
  if (conv.referee) {
    const hidden = $("messages").classList.contains("hide-referee");
    controls.appendChild(btn(hidden ? "show referee" : "hide referee", "", () => {
      $("messages").classList.toggle("hide-referee");
      renderHeader();
    }));
  }
  const showThinking = $("messages").classList.contains("show-thinking");
  controls.appendChild(btn(showThinking ? "hide thinking" : "show thinking", "", () => {
    $("messages").classList.toggle("show-thinking");
    renderHeader();
  }));
  controls.appendChild(btn("edit", "", openEditModal));
  controls.appendChild(btn("clear", "danger", () => {
    active.ws.send(JSON.stringify({ type: "clear_messages" }));
  }));

  $("composer").hidden = !conv.participants.some((p) => p.kind === "user");
  $("composer").onsubmit = (e) => {
    e.preventDefault();
    const text = $("input").value.trim();
    if (!text) return;
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
  if (avatar) return;
  avatar = new Avatar($("avatar-canvas"));
  // Wire mic + audio immediately — don't block on the .glb network fetch.
  audioQueue = new AudioQueue(avatar);
  ptt = new PushToTalk({
    button: $("mic-btn"),
    onPressStart: () => { if (avatar) avatar.stopSpeaking(); },
    onTranscript: (text) => {
      if (!active) return;
      active.ws.send(JSON.stringify({ type: "user_message", content: text }));
    },
  });
  $("stop-btn").onclick = () => { if (avatar) avatar.stopSpeaking(); };
  // Load the avatar mesh in the background so failures don't break voice.
  avatar.load().catch((e) => {
    console.error("avatar load failed:", e);
    $("mic-status").textContent = `avatar load failed: ${e.message}`;
  });
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
  if (m.sender === -1) {
    el.classList.add("referee");
    who = "referee";
  } else {
    const p = conv.participants[m.sender];
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
  fillModelSelect($("f-ref-model"));
  $("f-cancel").onclick = closeModal;
  $("f-create").onclick = createConv;
  $("edit-cancel").onclick = () => { $("edit-modal").hidden = true; };
  $("edit-save").onclick = saveEditModal;
  $("f-mode").onchange = renderParticipantFields;
  $("f-avatar-mode").onchange = renderParticipantFields;
  $("f-referee-on").onchange = () => {
    $("f-ref-model").disabled = !$("f-referee-on").checked;
    $("f-ref-cadence").disabled = !$("f-referee-on").checked;
    $("f-ref-intervene").disabled = !$("f-referee-on").checked;
    $("f-ref-system").disabled = !$("f-referee-on").checked;
  };
}

function openModal() {
  $("modal").hidden = false;
  renderParticipantFields();
  $("f-referee-on").onchange();
}
function closeModal() { $("modal").hidden = true; }

function fillModelSelect(sel) {
  sel.innerHTML = "";
  for (const m of models) {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    sel.appendChild(o);
  }
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
  if ($("f-referee-on").checked) {
    body.referee = {
      model: $("f-ref-model").value,
      system: $("f-ref-system").value,
      cadence: parseInt($("f-ref-cadence").value || "2", 10),
      intervene: $("f-ref-intervene").checked,
    };
  }

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

  if (conv.referee) {
    const card = document.createElement("div");
    card.className = "participant-card";
    card.dataset.ref = "1";
    card.innerHTML = `
      <h3>referee</h3>
      <label>model <select data-k="model"></select></label>
      <label>cadence <input data-k="cadence" type="number" min="1" /></label>
      <label><input data-k="intervene" type="checkbox" /> inject into participants</label>
      <label>system prompt <textarea data-k="system" rows="5"></textarea></label>
    `;
    const sel = card.querySelector("select");
    fillModelSelect(sel);
    sel.value = conv.referee.model || "";
    card.querySelector('[data-k=cadence]').value = conv.referee.cadence;
    card.querySelector('[data-k=intervene]').checked = !!conv.referee.intervene;
    card.querySelector('[data-k=system]').value = conv.referee.system || "";
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
    if (card.dataset.ref === "1") {
      patch.referee = {
        model: card.querySelector('[data-k=model]').value,
        system: card.querySelector('[data-k=system]').value,
        cadence: parseInt(card.querySelector('[data-k=cadence]').value, 10),
        intervene: card.querySelector('[data-k=intervene]').checked,
      };
    } else {
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
