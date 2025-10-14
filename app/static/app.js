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
  if (!file) {
    alert("Please select a video file first.");
    return;
  }

  const model = modelSelect.value;
  const customFileInput = document.getElementById("modelFile");
  const customModelFile = model === "custom" ? customFileInput.files[0] : null;

  if (model === "custom" && !customModelFile) {
    alert("Please upload your custom YOLO .pt model file.");
    return;
  }

  const fd = new FormData();
  fd.append("file", file);
  fd.append("model", model);
  if (customModelFile) fd.append("custom_model", customModelFile);

  log("Uploading files...");
  const resp = await fetch("/upload", { method: "POST", body: fd });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    alert("Upload failed: " + (err.detail || resp.statusText));
    return;
  }

  const data = await resp.json();
  jobId = data.job_id;
  const wsPath = data.ws;

  statusDiv.classList.remove("hidden");
  metricsEl.textContent = "Processing started. Connecting WebSocket...";
  openWs(wsPath);
});

function openWs(wsPath) {
  const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + wsPath;
  ws = new WebSocket(url);

  ws.onopen = () => {
    log("✅ WebSocket connected");
    metricsEl.textContent = "Connected. Receiving frames...";
  };

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      handleMsg(msg);
    } catch (e) {
      console.warn("Non-JSON message:", e);
    }
  };

  ws.onclose = () => log("❌ WebSocket closed");
  ws.onerror = (e) => console.error("WebSocket error:", e);
}

function handleMsg(msg) {
  if (msg.type === "info") {
    log("ℹ️ " + msg.message);
  } else if (msg.type === "progress") {
    const { frame = 0, total_frames = 0, pct = 0 } = msg;
    progressBar.style.width = pct + "%";
    progressInfo.textContent = `${frame} / ${total_frames} (${pct}%)`;
  } else if (msg.type === "frame") {
    liveFrame.src = "data:image/jpeg;base64," + msg.data;
  } else if (msg.type === "done") {
    metricsEl.textContent = JSON.stringify(msg.metrics, null, 2);
    downloadBtn.disabled = false;
    log("✅ Inference complete");
  } else if (msg.type === "error") {
    alert("Error: " + msg.message);
    metricsEl.textContent = "Error: " + msg.message;
  }
}

downloadBtn.addEventListener("click", async () => {
  if (!jobId) return;
  downloadBtn.disabled = true;
  downloadBtn.textContent = "Preparing...";

  const resp = await fetch(`/download/${jobId}`);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
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
  log("💾 Download finished");
});
