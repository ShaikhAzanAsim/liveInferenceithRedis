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
let lastCustomModelName = null; // ✅ store uploaded custom model filename

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

// === Custom Model Analysis Popup ===
const classPopup = document.getElementById("class-popup");
const classListDiv = document.getElementById("class-list");
const resetBtn = document.getElementById("reset-colors");
const saveBtn = document.getElementById("save-colors");

let classColors = {};

async function analyzeCustomModel(file) {
  const fd = new FormData();
  fd.append("model_file", file);
  lastCustomModelName = file.name; // ✅ remember the uploaded model filename

  const resp = await fetch("/analyze_model", { method: "POST", body: fd });
  if (!resp.ok) {
    const err = await resp.json();
    alert("Failed to analyze model: " + (err.detail || resp.statusText));
    return;
  }

  const data = await resp.json();
  const classNames = data.class_names;
  showClassPopup(classNames);
}

function showClassPopup(classNames) {
  classListDiv.innerHTML = "";
  classColors = {};

  classNames.forEach(name => {
    const color = getRandomColor();
    classColors[name] = color;

    const div = document.createElement("div");
    div.className = "class-item";
    div.innerHTML = `
      <span>${name}</span>
      <input type="color" value="${color}" data-class="${name}" />
    `;
    classListDiv.appendChild(div);
  });

  classPopup.classList.remove("hidden");
}

resetBtn.addEventListener("click", () => {
  document.querySelectorAll('#class-list input[type="color"]').forEach(input => {
    const randomColor = getRandomColor();
    input.value = randomColor;
    classColors[input.dataset.class] = randomColor;
  });
});

saveBtn.addEventListener("click", async () => {
  // Gather the current colors from all inputs
  document.querySelectorAll('#class-list input[type="color"]').forEach(input => {
    classColors[input.dataset.class] = input.value;
  });

  // Save locally for persistence
  localStorage.setItem("customModelColors", JSON.stringify(classColors));

  // 🎯 Send to backend with model_name included
  try {
    const response = await fetch("/set_class_colors", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_name: lastCustomModelName || "custom_model.pt", // ✅ include model name
        colors: classColors
      })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      alert("Failed to apply colors: " + (err.detail || response.statusText));
      return;
    }

    const resData = await response.json();
    console.log("🎨 Colors updated on backend:", resData);
    alert("✅ Colors saved! They’ll be used immediately in your next inference.");
    classPopup.classList.add("hidden");
  } catch (err) {
    console.error("Error sending colors:", err);
    alert("⚠️ Could not save colors to backend.");
  }
});

function getRandomColor() {
  const letters = "0123456789ABCDEF";
  let color = "#";
  for (let i = 0; i < 6; i++) color += letters[Math.floor(Math.random() * 16)];
  return color;
}

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

modelSelect.addEventListener("change", async () => {
  const customModelContainer = document.getElementById("custom-model-container");
  const showCustom = modelSelect.value === "custom";
  customModelContainer.classList.toggle("hidden", !showCustom);

  if (showCustom) {
    document.getElementById("modelFile").addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (file) {
        await analyzeCustomModel(file);
      }
    });
  }
});
