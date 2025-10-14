const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file");
const modelSelect = document.getElementById("model-select");
const statusDiv = document.getElementById("status");
const progressBar = document.getElementById("progress-bar");
const progressInfo = document.getElementById("progress-info");
const liveFrame = document.getElementById("live-frame");
const metricsEl = document.getElementById("metrics");
const downloadBtn = document.getElementById("download-btn");
const logDiv = document.getElementById("log");

let jobId = null;
let ws = null;

function log(msg) {
  const p = document.createElement("div");
  p.textContent = msg;
  logDiv.prepend(p);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) { alert("Select a video file."); return; }
  const model = modelSelect.value;

  // Upload via fetch
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model", model);

  const resp = await fetch("/upload", { method: "POST", body: fd });
  if (!resp.ok) {
    const err = await resp.json();
    alert("Upload failed: " + (err.detail || resp.statusText));
    return;
  }
  const data = await resp.json();
  jobId = data.job_id;
  const wsPath = "/ws/jobs/" + jobId;

  statusDiv.classList.remove("hidden");
  metricsEl.textContent = "Processing started, connecting WS...";
  openWs(wsPath);
});

function openWs(wsPath) {
  const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + wsPath;
  ws = new WebSocket(url);

  ws.onopen = () => {
    log("WS connected");
    metricsEl.textContent = "Connected. Waiting frames...";
  };
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      handleMsg(msg);
    } catch (e) {
      console.warn("Non-JSON message", e);
    }
  };
  ws.onclose = () => { log("WS closed"); };
  ws.onerror = (e) => { console.error(e); };
}

function handleMsg(msg) {
  if (msg.type === "info") {
    log("INFO: " + msg.message);
  } else if (msg.type === "progress") {
    const frame = msg.frame || 0;
    const total = msg.total_frames || 0;
    const pct = msg.pct || 0;
    progressBar.style.width = pct + "%";
    progressInfo.textContent = `${frame} / ${total} (${pct}%)`;
  } else if (msg.type === "frame") {
    // contains base64 data
    const b64 = msg.data;
    liveFrame.src = "data:image/jpeg;base64," + b64;
  } else if (msg.type === "done") {
    metricsEl.textContent = JSON.stringify(msg.metrics, null, 2);
    downloadBtn.disabled = false;
    log("Inference done");
  } else if (msg.type === "error") {
    alert("Error: " + msg.message);
    metricsEl.textContent = "Error: " + msg.message;
  }
}

downloadBtn.addEventListener("click", async () => {
  if (!jobId) return;
  downloadBtn.disabled = true;
  downloadBtn.textContent = "Preparing ...";
  const resp = await fetch(`/download/${jobId}`);
  if (!resp.ok) {
    const err = await resp.json();
    alert("Download failed: " + (err.detail || resp.statusText));
    downloadBtn.disabled = false;
    downloadBtn.textContent = "Download Video";
    return;
  }
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${jobId}.mp4`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  downloadBtn.textContent = "Downloaded";
  log("Download finished");
});
