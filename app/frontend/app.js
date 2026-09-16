const visual = document.querySelector("#visual-stage");
const preview = document.querySelector("#preview");
const cameraButton = document.querySelector("#camera-button");
const predictButton = document.querySelector("#predict-button");
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

let pendingFile = null;
let mediaStream = null;

function setBadge(state, text) {
  resultStatus.textContent = text;
  resultStatus.className = "badge " + state;
}

function showWarnings(list) {
  if (list && list.length) {
    warnings.textContent = list.join("；");
    warnings.hidden = false;
  } else {
    warnings.hidden = true;
  }
}

async function loadHealth() {
  try {
    const health = await (await fetch("/health")).json();
    if (health.model_ready) {
      systemStatus.textContent = "模型已就绪，可进行识别";
      systemStatus.dataset.kind = "ready";
    } else if (health.demo_mode) {
      systemStatus.textContent = "界面演示模式：结果非模型预测，不用于实验报告";
      systemStatus.dataset.kind = "warning";
    } else {
      systemStatus.textContent = "模型尚未安装，仅可检查界面与上传流程";
      systemStatus.dataset.kind = "warning";
    }
  } catch (error) {
    systemStatus.textContent = "无法连接后端：" + error.message;
    systemStatus.dataset.kind = "error";
  }
}

cameraButton.addEventListener("click", async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    preview.srcObject = mediaStream;
    visual.classList.add("has-video");
    cameraButton.textContent = "摄像头已开启";
    recordButton.disabled = false;
    pendingFile = null;
    predictButton.disabled = true;
    selection.textContent = "摄像头已就绪，可点击「录制 3 秒」";
  } catch (error) {
    selection.textContent = "无法开启摄像头：" + error.message;
  }
});

recordButton.addEventListener("click", () => {
  selection.textContent = "录制功能需要后端支持，请使用视频文件上传。";
});

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  pendingFile = file;
  const url = URL.createObjectURL(file);
  preview.src = url;
  visual.classList.add("has-video");
  selection.textContent = "已选择：" + file.name + "，点击「识别」开始";
  predictButton.disabled = false;
  setBadge("idle", "等待识别");
  resultText.textContent = "已选择视频，点击「识别」按钮进行预测。";
  showWarnings(null);
});

async function predictOnce() {
  setBadge("running", "识别中…");
  resultText.textContent = "正在识别，请稍候…";
  try {
    const formData = new FormData();
    formData.append("video", pendingFile);

    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();

    if (data.error) {
      setBadge("fail", "识别失败");
      resultText.textContent = "识别失败";
      showWarnings([data.error]);
      return;
    }

    resultIntent.textContent = data.intent || "unknown";
    resultGloss.textContent = data.gloss_sequence || data.gloss || "-";
    resultText.textContent = data.text_zh || "-";
    resultConfidence.textContent = data.confidence
      ? (data.confidence * 100).toFixed(1) + "%"
      : "-";
    resultLatency.textContent = data.latency_ms?.total
      ? data.latency_ms.total.toFixed(1) + "ms"
      : "-";
    setBadge(data.status === "ok" ? "done" : "fail", data.status === "ok" ? "识别完成" : "识别失败");
    showWarnings(data.warnings);
  } catch (error) {
    setBadge("fail", "识别失败");
    resultText.textContent = "请求失败";
    showWarnings([error.message]);
  }
}

predictButton.addEventListener("click", predictOnce);

loadHealth();