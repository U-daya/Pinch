// Pinch: everything that runs in the browser.
//
// The pipeline, same shape as the original:
//   webcam -> MediaPipe finds 21 hand points -> turn points into features ->
//   classify the gesture (rules now, a small neural network in Stage 2) -> act on the canvas
//
// And in parallel:
//   your voice -> text -> server.py -> Gemini picks a tool -> this file draws it

import { FilesetResolver, HandLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const board = $("board"), ink = $("ink"), items = $("items"), cursor = $("cursor");
const video = $("video"), overlay = $("overlay");
const inkCtx = ink.getContext("2d"), overlayCtx = overlay.getContext("2d");

// ---------------------------------------------------------------------------
// 1. Webcam + MediaPipe hand tracking
// ---------------------------------------------------------------------------
let landmarker;

async function startCamera() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
    });
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    await video.play();
    setPill("camStatus", "camera: on", true);
    requestAnimationFrame(loop);
  } catch (error) {
    setPill("camStatus", "camera: off (" + error.name + ")");
    console.error(error);
  }
}

let lastVideoTime = -1;
function loop() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    const hand = result.landmarks?.[0];
    drawHandPreview(hand);
    if (hand) onHand(mirror(hand)); else onNoHand();
  }
  requestAnimationFrame(loop);
}

// The preview is mirrored like a selfie, so flip x to match what you see
const mirror = (hand) => hand.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z }));

// ---------------------------------------------------------------------------
// 2. Features: turn 21 points into numbers that do not depend on where your hand is
//    or how close it is to the camera. This is the "feature engineering" step the
//    original team described (distances and positions between key points).
// ---------------------------------------------------------------------------
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const TIPS = [4, 8, 12, 16, 20]; // thumb, index, middle, ring, pinky fingertips

function features(hand) {
  const wrist = hand[0];
  const size = dist(hand[0], hand[9]) || 1e-6; // wrist to middle knuckle = hand size
  const out = [];
  for (const p of hand) out.push((p.x - wrist.x) / size, (p.y - wrist.y) / size); // 42 numbers
  for (let i = 0; i < TIPS.length; i++)                                              // 10 numbers
    for (let j = i + 1; j < TIPS.length; j++) out.push(dist(hand[TIPS[i]], hand[TIPS[j]]) / size);
  return out; // 52 features total
}

// ---------------------------------------------------------------------------
// 3. Gesture classifier. Stage 1 = hand-written rules. Stage 2 = trained network.
// ---------------------------------------------------------------------------
function ruleClassify(hand) {
  const wrist = hand[0];
  const size = dist(hand[0], hand[9]);
  // A finger counts as "up" when its tip is farther from the wrist than its middle joint
  const up = [[8, 6], [12, 10], [16, 14], [20, 18]].map(([tip, pip]) =>
    dist(hand[tip], wrist) > dist(hand[pip], wrist) * 1.1);
  const [index, middle, ring, pinky] = up;
  const pinch = dist(hand[4], hand[8]) / size < 0.3;

  if (pinch) return "pinch";
  if (index && middle && ring && pinky) return "palm";
  if (!index && !middle && !ring && !pinky) return "fist";
  if (index && middle) return "hover";
  if (index) return "draw";
  return "hover";
}

let model = null; // filled in if you trained one (see train_gestures.py)
async function loadModel() {
  try {
    const res = await fetch("/model");
    if (res.ok) { model = await res.json(); setPill("classifier", "neural network", true); }
  } catch { /* no model yet, rules it is */ }
}

function modelClassify(hand) {
  // A tiny neural network forward pass: scale inputs, then (weights x input + bias) per layer
  let x = features(hand).map((v, i) => (v - model.mean[i]) / model.scale[i]);
  model.layers.forEach((layer, n) => {
    x = layer.b.map((b, j) => b + x.reduce((sum, xi, i) => sum + xi * layer.W[i][j], 0));
    if (n < model.layers.length - 1) x = x.map((v) => Math.max(0, v)); // ReLU on hidden layers
  });
  return model.labels[x.indexOf(Math.max(...x))];
}

const classify = (hand) => (model ? modelClassify(hand) : ruleClassify(hand));

// Only accept a gesture after it shows up 3 frames in a row. Stops flickering.
let candidate = "none", candidateFrames = 0, gesture = "none";
function stable(label) {
  if (label === candidate) candidateFrames++; else { candidate = label; candidateFrames = 1; }
  if (candidateFrames >= 3) gesture = candidate;
  return gesture;
}

// ---------------------------------------------------------------------------
// 4. Act on the gesture
// ---------------------------------------------------------------------------
let pos = null, lastInk = null, palmSince = 0, grabbed = null, grabOffset = null, pinchStart = 0;

function onHand(hand) {
  recordSample(hand);
  const g = stable(classify(hand));
  setPill("gesture", "gesture: " + g, g !== "none");

  // Map the middle 80% of the camera view to the whole board, so you can reach the edges
  const tip = hand[8];
  const rect = board.getBoundingClientRect();
  const target = {
    x: clamp((tip.x - 0.1) / 0.8, 0, 1) * rect.width,
    y: clamp((tip.y - 0.1) / 0.8, 0, 1) * rect.height,
  };
  // Smooth the cursor so it does not jitter (moving average)
  pos = pos ? { x: pos.x * 0.6 + target.x * 0.4, y: pos.y * 0.6 + target.y * 0.4 } : target;
  cursor.style.display = "block";
  cursor.style.left = pos.x + "px";
  cursor.style.top = pos.y + "px";
  cursor.className = g;

  // Draw
  if (g === "draw") {
    if (lastInk) {
      inkCtx.beginPath(); inkCtx.moveTo(lastInk.x, lastInk.y); inkCtx.lineTo(pos.x, pos.y); inkCtx.stroke();
    }
    lastInk = { ...pos };
  } else lastInk = null;

  // Open palm held 1.5 seconds clears your drawing
  if (g === "palm") {
    palmSince ||= performance.now();
    if (performance.now() - palmSince > 1500) { clearInk(); palmSince = 0; }
  } else palmSince = 0;

  // Pinch: grab and move a card, or flip a flashcard with a quick pinch
  if (g === "pinch" && !grabbed) {
    const el = itemAt(pos);
    if (el) {
      grabbed = el; pinchStart = performance.now();
      grabOffset = { x: pos.x - el.offsetLeft, y: pos.y - el.offsetTop };
      el.classList.add("grabbed");
    }
  } else if (g === "pinch" && grabbed) {
    grabbed.style.left = pos.x - grabOffset.x + "px";
    grabbed.style.top = pos.y - grabOffset.y + "px";
  } else if (g !== "pinch" && grabbed) {
    const quick = performance.now() - pinchStart < 400;
    if (quick && grabbed.classList.contains("flashcard")) flip(grabbed);
    grabbed.classList.remove("grabbed");
    grabbed = null;
  }
}

function onNoHand() {
  cursor.style.display = "none";
  lastInk = null;
  stable("none");
  setPill("gesture", "gesture: none");
}

function itemAt(p) {
  const rect = board.getBoundingClientRect();
  const el = document.elementFromPoint(rect.left + p.x, rect.top + p.y);
  return el?.closest(".item");
}

// ---------------------------------------------------------------------------
// 5. Ink canvas
// ---------------------------------------------------------------------------
function resizeInk() {
  const saved = inkCtx.getImageData(0, 0, ink.width, ink.height);
  ink.width = board.clientWidth; ink.height = board.clientHeight;
  inkCtx.putImageData(saved, 0, 0);
  inkCtx.lineWidth = 4; inkCtx.lineCap = "round"; inkCtx.strokeStyle = "#1f2a37";
}
const clearInk = () => inkCtx.clearRect(0, 0, ink.width, ink.height);

// ---------------------------------------------------------------------------
// 6. Voice + text -> server.py -> Gemini tool calls
// ---------------------------------------------------------------------------
async function askAgent(text) {
  if (!text.trim()) return;
  setPill("agentMode", "agent: thinking...", true);
  try {
    const res = await fetch("/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, material: $("material").value }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    data.actions.forEach(runAction);
    setPill("agentMode", "agent: " + data.mode);
  } catch (error) {
    setPill("agentMode", "agent: error");
    console.error(error);
    addNote({ title: "Agent error", text: String(error.message || error) });
  }
}

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognizer = null;
function toggleMic() {
  if (!SpeechRecognition) return alert("Voice needs Chrome. You can type instead.");
  if (recognizer) { recognizer.stop(); return; }
  recognizer = new SpeechRecognition();
  recognizer.lang = "en-US";
  recognizer.onresult = (e) => { const said = e.results[0][0].transcript; $("textInput").value = said; askAgent(said); };
  recognizer.onend = () => { recognizer = null; $("mic").classList.remove("listening"); };
  recognizer.start();
  $("mic").classList.add("listening");
}

// ---------------------------------------------------------------------------
// 7. Tools: each one matches a tool declared in server.py
// ---------------------------------------------------------------------------
const ACTIONS = {
  add_note: addNote,
  draw_concept_map: drawConceptMap,
  draw_chart: drawChart,
  make_flashcards: makeFlashcards,
  clear_canvas: () => { items.innerHTML = ""; clearInk(); slot = 0; },
};

function runAction({ name, args }) {
  (ACTIONS[name] || ((a) => addNote({ title: "Unknown tool " + name, text: JSON.stringify(a) })))(args || {});
}

// Place new items in a simple grid so they do not pile on top of each other
let slot = 0;
function place(el) {
  const cols = Math.max(1, Math.floor((board.clientWidth - 340) / 300));
  el.style.left = 330 + (slot % cols) * 300 + "px";
  el.style.top = 20 + Math.floor(slot / cols) * 230 + "px";
  slot++;
  items.appendChild(el);
  makeMouseDraggable(el);
  return el;
}

function addNote({ title = "", text = "" }) {
  const el = document.createElement("div");
  el.className = "item note";
  el.innerHTML = `<h4></h4><div></div>`;
  el.querySelector("h4").textContent = title;
  el.querySelector("div").textContent = text;
  place(el);
}

function makeFlashcards({ cards = [] }) {
  cards.forEach((card) => {
    const el = document.createElement("div");
    el.className = "item flashcard";
    el.dataset.front = card.front; el.dataset.back = card.back;
    el.innerHTML = `<span></span><small>pinch to flip</small>`;
    el.querySelector("span").textContent = card.front;
    el.addEventListener("dblclick", () => flip(el));
    place(el);
  });
}

function flip(el) {
  el.classList.toggle("flipped");
  el.querySelector("span").textContent = el.classList.contains("flipped") ? el.dataset.back : el.dataset.front;
}

function drawConceptMap({ center = "", branches = [] }) {
  const W = 520, H = 400, cx = W / 2, cy = H / 2;
  const svgNode = (x, y, text, big) => {
    const w = Math.min(160, 14 + text.length * (big ? 8.5 : 7));
    return `<rect x="${x - w / 2}" y="${y - 15}" width="${w}" height="30" rx="15"
              fill="${big ? "#2f6fed" : "#eef3ff"}" stroke="#2f6fed"/>
            <text x="${x}" y="${y + 5}" text-anchor="middle" font-size="${big ? 14 : 12}"
              fill="${big ? "#fff" : "#1f2a37"}">${escape(text)}</text>`;
  };
  let lines = "", nodes = "";
  branches.forEach((branch, i) => {
    const angle = (i / branches.length) * Math.PI * 2 - Math.PI / 2;
    const bx = cx + Math.cos(angle) * 115, by = cy + Math.sin(angle) * 105;
    lines += `<line x1="${cx}" y1="${cy}" x2="${bx}" y2="${by}" stroke="#9db6f5" stroke-width="2"/>`;
    const kids = branch.children || [];
    kids.forEach((child, k) => {
      const a = angle + (k - (kids.length - 1) / 2) * 0.42;
      const kx = cx + Math.cos(a) * 200, ky = cy + Math.sin(a) * 170;
      lines += `<line x1="${bx}" y1="${by}" x2="${kx}" y2="${ky}" stroke="#c9d6f7" stroke-width="1.5"/>`;
      nodes += svgNode(kx, ky, child, false);
    });
    nodes += svgNode(bx, by, branch.label, false);
  });
  nodes += svgNode(cx, cy, center, true);

  const el = document.createElement("div");
  el.className = "item";
  el.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${lines}${nodes}</svg>`;
  place(el);
}

function drawChart({ chart_type = "bar", title = "", labels = [], values = [] }) {
  const el = document.createElement("div");
  el.className = "item";
  el.innerHTML = `<h4></h4><canvas width="360" height="220"></canvas>`;
  el.querySelector("h4").textContent = title;
  const ctx = el.querySelector("canvas").getContext("2d");
  const pad = 30, w = 360 - pad * 2, h = 220 - pad * 2;
  const max = Math.max(...values, 1);
  ctx.strokeStyle = "#d9d3c4"; ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, pad + h); ctx.lineTo(pad + w, pad + h); ctx.stroke();
  ctx.font = "11px system-ui"; ctx.fillStyle = "#6b7280"; ctx.textAlign = "center";
  const step = w / Math.max(values.length, 1);
  const points = values.map((v, i) => ({ x: pad + step * (i + 0.5), y: pad + h - (v / max) * h, v }));
  ctx.fillStyle = "#2f6fed"; ctx.strokeStyle = "#2f6fed"; ctx.lineWidth = 2;
  if (chart_type === "line") {
    ctx.beginPath(); points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.stroke();
    points.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2); ctx.fill(); });
  } else {
    points.forEach((p) => ctx.fillRect(p.x - step * 0.3, p.y, step * 0.6, pad + h - p.y));
  }
  ctx.fillStyle = "#1f2a37";
  points.forEach((p, i) => {
    ctx.fillText(String(labels[i] ?? ""), p.x, pad + h + 14);
    ctx.fillText(String(p.v), p.x, p.y - 6);
  });
  place(el);
}

// ---------------------------------------------------------------------------
// 8. Stage 2 helper: record labeled gesture samples to train a neural network
// ---------------------------------------------------------------------------
const samples = [];
let recording = false;
window.addEventListener("keydown", (e) => { if (e.key === "r" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") recording = true; });
window.addEventListener("keyup", (e) => { if (e.key === "r") recording = false; });

function recordSample(hand) {
  if (!recording) return;
  samples.push({ label: $("recLabel").value, features: features(hand) });
  $("recCount").textContent = samples.length + " samples";
}

function downloadSamples() {
  const blob = new Blob([JSON.stringify(samples)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "gestures.json"; a.click();
}

// ---------------------------------------------------------------------------
// Small helpers and wiring
// ---------------------------------------------------------------------------
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const escape = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function setPill(id, text, on = false) { $(id).textContent = text; $(id).classList.toggle("on", on); }

function drawHandPreview(hand) {
  overlay.width = overlay.clientWidth; overlay.height = overlay.clientHeight;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (!hand) return;
  overlayCtx.fillStyle = "#4ade80";
  for (const p of hand) {
    overlayCtx.beginPath();
    overlayCtx.arc((1 - p.x) * overlay.width, p.y * overlay.height, 3, 0, Math.PI * 2);
    overlayCtx.fill();
  }
}

// Mouse fallback so you can also drag cards without a camera
function makeMouseDraggable(el) {
  el.addEventListener("pointerdown", (e) => {
    const startX = e.clientX - el.offsetLeft, startY = e.clientY - el.offsetTop;
    const move = (ev) => { el.style.left = ev.clientX - startX + "px"; el.style.top = ev.clientY - startY + "px"; };
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  });
}

$("send").onclick = () => askAgent($("textInput").value);
$("textInput").addEventListener("keydown", (e) => { if (e.key === "Enter") askAgent(e.target.value); });
$("mic").onclick = toggleMic;
$("materialBtn").onclick = () => $("materialDialog").showModal();
$("recDownload").onclick = downloadSamples;
window.addEventListener("resize", resizeInk);

resizeInk();
loadModel();
startCamera();
