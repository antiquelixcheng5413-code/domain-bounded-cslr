const preview = document.querySelector("#preview");
const cameraButton = document.querySelector("#camera-button");
const recordButton = document.querySelector("#record-button");
const fileInput = document.querySelector("#file-input");
const selection = document.querySelector("#selection");
const systemStatus = document.querySelector("#system-status");
const resultStatus = document.querySelector("#result-status");
const resultText = document.querySelector("#result-text");
const resultIntent = document.querySelector("#result-intent");
const resultGloss = document.querySelector("#result-gloss");
const resultConfidence = document.querySelector("#result-confidence");
const resultLatency = document.querySelector("#result-latency");
const warnings = document.querySelector("#warnings");

let mediaStream;

async function loadHealth() {
  try {
    const response = await fetch("/api/v1/health");
    const health = await response.json();
    if (health.model_ready) {
      systemStatus.textContent = "模型已加载，可以进行真实识别。";
      systemStatus.dataset.kind = "ready";
    } else if (health.demo_mode) {
      systemStatus.textContent = "界面演示模式：结果不是模型预测，不能用于实验报告。";
      systemStatus.dataset.kind = "warning";
    } else {
      systemStatus.textContent = "模型尚未安装。可检查界面和上传流程，但不能进行真实识别。";
      systemStatus.dataset.kind = "warning";
    }
  } catch (error) {
    systemStatus.textContent = `无法连接后端：${error.message}`;
    systemStatus.dataset.kind = "error";
  }
}

cameraButton.addEventListener("click", async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    preview.srcObject = mediaStream;
    recordButton.disabled = false;
    cameraButton.textContent = "摄像头已开启";
  } catch (error) {
    selection.textContent = `无法开启摄像头：${error.message}`;
  }
});

recordButton.addEventListener("click", async () => {
  if (!mediaStream) return;
  recordButton.disabled = true;
  selection.textContent = "正在录制 3 秒...";
  const options = MediaRecorder.isTypeSupported("video/webm") ? { mimeType: "video/webm" } : {};
  const recorder = new MediaRecorder(mediaStream, options);
  const chunks = [];
  recorder.addEventListener("dataavailable", (event) => chunks.push(event.data));
  const completed = new Promise((resolve) => recorder.addEventListener("stop", resolve));
  recorder.start();
  window.setTimeout(() => recorder.stop(), 3000);
  await completed;
  const mimeType = recorder.mimeType || "video/webm";
  const blob = new Blob(chunks, { type: mimeType });
  const file = new File([blob], "camera.webm", { type: mimeType });
  selection.textContent = "录制完成，正在上传...";
  await submitVideo(file);
  recordButton.disabled = false;
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  preview.srcObject = null;
  preview.src = URL.createObjectURL(file);
  preview.controls = true;
  selection.textContent = `${file.name}，正在上传...`;
  await submitVideo(file);
});

async function submitVideo(file) {
  const payload = new FormData();
  payload.append("video", file, file.name);
  resultStatus.textContent = "处理中";
  resultText.textContent = "正在提取特征并执行推理...";
  warnings.replaceChildren();

  try {
    const response = await fetch("/api/v1/predict", { method: "POST", body: payload });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "请求失败");
    renderResult(body);
    selection.textContent = `${file.name}，${formatBytes(file.size)}`;
  } catch (error) {
    resultStatus.textContent = "错误";
    resultText.textContent = error.message;
    resultIntent.textContent = "-";
    resultGloss.textContent = "-";
    resultConfidence.textContent = "-";
    resultLatency.textContent = "-";
  }
}

function renderResult(result) {
  resultStatus.textContent = result.status;
  resultText.textContent = result.text_zh;
  resultIntent.textContent = result.intent;
  resultGloss.textContent = result.gloss;
  resultConfidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
  resultLatency.textContent = `${result.latency_ms.total ?? 0} ms`;
  warnings.replaceChildren(
    ...result.warnings.map((warning) => {
      const item = document.createElement("li");
      item.textContent = warning;
      return item;
    }),
  );
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

loadHealth();
