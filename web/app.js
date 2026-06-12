"use strict";

// ---------------------------------------------------------------------------
// Small DOM + API helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const SPEEDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 0.75];
const FLAGS = { pl: "🇵🇱", en: "🇬🇧" };

async function api(path, opts) {
  const res = await fetch(path, opts);
  return res;
}
async function apiJSON(path, opts) {
  const res = await api(path, opts);
  const data = await res.json().catch(() => ({}));
  return { res, data };
}

function fmtTime(sec) {
  if (!isFinite(sec) || sec < 0) sec = 0;
  sec = Math.floor(sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  const mm = h ? String(m).padStart(2, "0") : String(m);
  const ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let voices = { pl: [], en: [] };
let pollTimer = null;
let currentJobId = null; // job loaded in the player

// ---------------------------------------------------------------------------
// Input card: language / voice / char count
// ---------------------------------------------------------------------------
const textEl = $("text");
const langEl = $("lang");
const voiceEl = $("voice");
const voiceWrap = $("voice-wrap");

textEl.addEventListener("input", () => {
  $("charcount").textContent = `${textEl.value.length.toLocaleString()} characters`;
});

function updateVoiceUI() {
  const lang = langEl.value;
  if (lang === "auto") {
    voiceWrap.classList.add("hidden"); // voice falls back to default until detection
    return;
  }
  voiceWrap.classList.remove("hidden");
  const list = voices[lang] || [];
  voiceEl.innerHTML = "";
  for (const v of list) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    voiceEl.appendChild(opt);
  }
}
langEl.addEventListener("change", updateVoiceUI);

async function loadVoices() {
  try {
    const { data } = await apiJSON("/api/voices");
    voices = { pl: data.pl || [], en: data.en || [] };
  } catch (_) {
    voices = { pl: [], en: [] };
  }
  updateVoiceUI();
}

// ---------------------------------------------------------------------------
// Generate + poll
// ---------------------------------------------------------------------------
const genBtn = $("generate");
const progressEl = $("progress");
const barFill = $("bar-fill");
const progressLabel = $("progress-label");
const errorEl = $("error");

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}
function clearError() {
  errorEl.classList.add("hidden");
  errorEl.textContent = "";
}
function setGenerating(on) {
  genBtn.disabled = on;
  langEl.disabled = on;
  voiceEl.disabled = on;
  progressEl.classList.toggle("hidden", !on);
}

genBtn.addEventListener("click", async () => {
  clearError();
  const text = textEl.value.trim();
  if (!text) {
    showError("Paste some text first.");
    return;
  }
  const lang = langEl.value;
  const voice = lang === "auto" ? null : voiceEl.value || null;

  setGenerating(true);
  barFill.style.width = "0%";
  progressLabel.textContent = "Queued…";

  let job_id;
  try {
    const { res, data } = await apiJSON("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, lang, voice }),
    });
    if (res.status === 201 || res.status === 409) {
      job_id = data.job_id; // 409 = cache hit, same shape
    } else {
      throw new Error(data.detail || `Request failed (${res.status})`);
    }
  } catch (e) {
    setGenerating(false);
    showError(e.message || "Could not start generation.");
    return;
  }
  pollJob(job_id);
});

function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  const tick = async () => {
    let job;
    try {
      const { res, data } = await apiJSON(`/api/jobs/${jobId}`);
      if (!res.ok) throw new Error(data.detail || "lost the job");
      job = data;
    } catch (e) {
      clearInterval(pollTimer);
      setGenerating(false);
      showError(e.message);
      return;
    }
    renderProgress(job);
    if (job.status === "done") {
      clearInterval(pollTimer);
      setGenerating(false);
      loadJob(job);
      refreshLibrary();
    } else if (job.status === "failed") {
      clearInterval(pollTimer);
      setGenerating(false);
      showError("Generation failed: " + (job.error || "unknown error"));
      refreshLibrary();
    }
  };
  tick();
  pollTimer = setInterval(tick, 1500);
}

function renderProgress(job) {
  if (job.message) {
    progressLabel.textContent = job.message;
  } else if (job.status === "queued") {
    progressLabel.textContent = "Queued…";
  } else if (job.chunks_total) {
    progressLabel.textContent =
      `Generating… ${job.chunks_done}/${job.chunks_total} chunks`;
  } else {
    progressLabel.textContent = "Generating…";
  }
  barFill.style.width = `${Math.round((job.progress || 0) * 100)}%`;
}

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------
const audio = $("audio");
const playerCard = $("player-card");
const playPauseBtn = $("playpause");
const seek = $("seek");
const curEl = $("cur");
const durEl = $("dur");
const speedBtn = $("speed");
const sleepSel = $("sleep");
const sleepRemain = $("sleep-remain");

let speedIndex = 0;
let seeking = false;
let lastSaved = 0;
let sleepTimer = null;
let sleepEndsAt = 0;

function posKey(id) { return `pos:${id}`; }

function loadJob(job) {
  currentJobId = job.id;
  $("player-title").textContent = `${FLAGS[job.lang] || ""} ${job.title || "(untitled)"}`;
  playerCard.classList.remove("hidden");
  audio.src = `/api/audio/${job.id}.m4a`;
  audio.load();
  markActiveLibraryItem(job.id);
  // playbackRate persists across tracks; reflect it on the button.
  audio.playbackRate = SPEEDS[speedIndex];
}

audio.addEventListener("loadedmetadata", () => {
  durEl.textContent = fmtTime(audio.duration);
  const saved = parseFloat(localStorage.getItem(posKey(currentJobId)) || "0");
  if (saved > 1 && isFinite(audio.duration) && saved < audio.duration - 2) {
    audio.currentTime = saved;
  }
});

audio.addEventListener("timeupdate", () => {
  if (!seeking) {
    const d = audio.duration || 0;
    seek.value = d ? Math.round((audio.currentTime / d) * 1000) : 0;
    curEl.textContent = fmtTime(audio.currentTime);
  }
  const now = audio.currentTime;
  if (Math.abs(now - lastSaved) > 4 && currentJobId) {
    localStorage.setItem(posKey(currentJobId), String(now));
    lastSaved = now;
  }
});

audio.addEventListener("play", () => { playPauseBtn.textContent = "❚❚"; });
audio.addEventListener("pause", () => {
  playPauseBtn.textContent = "▶";
  if (currentJobId) localStorage.setItem(posKey(currentJobId), String(audio.currentTime));
});
audio.addEventListener("ended", () => {
  playPauseBtn.textContent = "▶";
  if (currentJobId) localStorage.removeItem(posKey(currentJobId));
});

function togglePlay() {
  if (audio.paused) audio.play(); else audio.pause();
}
function skip(delta) {
  const d = audio.duration || 0;
  audio.currentTime = Math.min(Math.max(0, audio.currentTime + delta), d || audio.currentTime + delta);
}

playPauseBtn.addEventListener("click", togglePlay);
$("back15").addEventListener("click", () => skip(-15));
$("fwd15").addEventListener("click", () => skip(15));

seek.addEventListener("input", () => {
  seeking = true;
  const d = audio.duration || 0;
  curEl.textContent = fmtTime((seek.value / 1000) * d);
});
seek.addEventListener("change", () => {
  const d = audio.duration || 0;
  audio.currentTime = (seek.value / 1000) * d;
  seeking = false;
});

speedBtn.addEventListener("click", () => {
  speedIndex = (speedIndex + 1) % SPEEDS.length;
  const rate = SPEEDS[speedIndex];
  audio.playbackRate = rate; // preservesPitch left at its (true) default
  speedBtn.textContent = `${rate}×`;
});

// Sleep timer
sleepSel.addEventListener("change", () => {
  const min = parseInt(sleepSel.value, 10);
  if (sleepTimer) { clearInterval(sleepTimer); sleepTimer = null; }
  if (!min) { sleepRemain.textContent = ""; return; }
  sleepEndsAt = Date.now() + min * 60000;
  const update = () => {
    const left = Math.round((sleepEndsAt - Date.now()) / 1000);
    if (left <= 0) {
      audio.pause();
      clearInterval(sleepTimer);
      sleepTimer = null;
      sleepSel.value = "0";
      sleepRemain.textContent = "";
      return;
    }
    sleepRemain.textContent = fmtTime(left);
  };
  update();
  sleepTimer = setInterval(update, 1000);
});

// Keyboard shortcuts (ignore while typing in a field)
document.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "textarea" || tag === "input" || tag === "select") return;
  if (playerCard.classList.contains("hidden")) return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  else if (e.code === "ArrowLeft") { e.preventDefault(); skip(-15); }
  else if (e.code === "ArrowRight") { e.preventDefault(); skip(15); }
});

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------
const libraryEl = $("library");

function markActiveLibraryItem(id) {
  for (const li of libraryEl.querySelectorAll(".lib-item")) {
    li.classList.toggle("active", li.dataset.id === id);
  }
}

async function refreshLibrary() {
  let jobs = [];
  try {
    const { data } = await apiJSON("/api/jobs");
    jobs = Array.isArray(data) ? data : [];
  } catch (_) { /* leave list as-is */ }

  libraryEl.innerHTML = "";
  if (!jobs.length) {
    libraryEl.innerHTML = '<li class="empty">No recordings yet.</li>';
    return;
  }
  for (const job of jobs) {
    libraryEl.appendChild(renderLibraryItem(job));
  }
  markActiveLibraryItem(currentJobId);
}

function renderLibraryItem(job) {
  const li = document.createElement("li");
  li.className = "lib-item";
  li.dataset.id = job.id;
  const playable = job.status === "done";
  if (playable) li.classList.add("playable");

  const flag = document.createElement("span");
  flag.className = "lib-flag";
  flag.textContent = FLAGS[job.lang] || "🌐";

  const mainDiv = document.createElement("div");
  mainDiv.className = "lib-main";
  const title = document.createElement("div");
  title.className = "lib-title";
  title.textContent = job.title || "(untitled)";
  const meta = document.createElement("div");
  meta.className = "lib-meta";
  const bits = [];
  if (job.duration_sec) bits.push(fmtTime(job.duration_sec));
  if (job.voice) bits.push(job.voice);
  bits.push(fmtDate(job.created_at));
  meta.textContent = bits.join(" · ");
  mainDiv.appendChild(title);
  mainDiv.appendChild(meta);

  const status = document.createElement("span");
  status.className = `lib-status ${job.status}`;
  status.textContent = job.status;

  const del = document.createElement("button");
  del.className = "lib-del";
  del.title = "Delete";
  del.textContent = "🗑";
  del.addEventListener("click", async (ev) => {
    ev.stopPropagation();
    await api(`/api/jobs/${job.id}`, { method: "DELETE" });
    if (currentJobId === job.id) {
      audio.pause();
      audio.removeAttribute("src");
      playerCard.classList.add("hidden");
      currentJobId = null;
    }
    refreshLibrary();
  });

  li.appendChild(flag);
  li.appendChild(mainDiv);
  li.appendChild(status);
  li.appendChild(del);

  if (playable) {
    li.addEventListener("click", () => loadJob(job));
  }
  return li;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
loadVoices();
refreshLibrary();
