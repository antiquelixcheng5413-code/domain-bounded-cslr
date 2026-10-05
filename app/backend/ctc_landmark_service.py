"""CTC landmark 识别服务（把 P42 训练的模型接进 FastAPI）

## 为什么需要这个

仓库原有的 `RecognitionService` 走 `artifacts/exports/lstm.onnx`，
而我们改进的是 **PyTorch CTC 模型**（P42，devWER 0.5211）。
两者架构不同，无法直接复用，故新写一个满足 `cslr.contracts.Prediction` 的服务。

## 契约

`cslr/contracts.py` 的 `Prediction`：
```
status, label, gloss_tokens, intent, gloss, text_zh,
confidence, top_k, warnings, latency_ms, model_version
```
前端 `app/frontend/app.js` 显示 `intent / gloss_sequence|gloss / text_zh /
confidence / latency_ms.total`。

## 关键实现要点

1. **词表必须与训练一致**：`build_ordered_vocabulary(train.csv, min_frequency=2,
   max_tokens=300)`。checkpoint 里存了 `vocab_size`，加载时校验。
2. **CTC 解码用仓库既有的 `decode_batch`**（P23 修复后与训练目标同空间），
   **不要自己写 argmax 折叠** —— 那正是 P20~P22 踩坑的地方。
3. **特征提取走 `realtime_landmark.py`**（mediapipe 0.10.35 新 API），
   仓库原提取器用已移除的 `mp.solutions.holistic`，那条路径跑不通。
4. **CTC 目标空间**：类 0 = blank，词表索引 i -> 类 i+1（P23 铁律）。
   模型的输出层是 `vocab_size + 1`。
5. **参考/置信度**：本服务做的是**词级识别**，没有 IntentCatalog 与中文翻译，
   故 `intent` 恒为 `unknown`、`text_zh` 用 gloss 拼接。**不伪造翻译结果。**
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# 🔴 三个路径都必须加（2026-10-05 修复）：
#   REPO/src            -> `cslr.*`
#   REPO/tools/blank_gov-> `p40_rgb_main` / `realtime_landmark`（平铺导入）
#   REPO（仓库根）      -> `tools.blank_gov.p40_rgb_main`
# 之前只加了前两个，`from tools.blank_gov.p40_rgb_main import DualInputCTC`
# 在调用方 path 顺序不同时抛 ModuleNotFoundError: No module named 'tools'，
# 被 _load() 的 try 吞掉 -> ready=False -> 前端一直 unavailable。
# 实测该失败间歇出现（取决于 sys.path 顺序），是「有时能识别有时不能」的根因。
for extra in (REPO, REPO / "src", REPO / "tools" / "blank_gov"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from cslr.contracts import Prediction  # noqa: E402


class CtcLandmarkService:
    """加载 P42 训练的 landmark CTC 模型并提供 predict_video。"""

    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        self.checkpoint = Path(checkpoint)
        self.model = None
        self.voc = None
        self.nrm = None
        self.extractor = None
        self.model_error: str | None = None
        self.demo_mode = False
        self.device = self._resolve_device(device)
        self._load()

    # ------------------------------------------------------------------ 设备
    @staticmethod
    def _resolve_device(name: str) -> str:
        if name != "auto":
            return name
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ 加载
    def _load(self) -> None:
        if not self.checkpoint.exists():
            self.model_error = f"checkpoint 不存在: {self.checkpoint}"
            return
        try:
            import torch

            from cslr.recognition.gloss_sequence import build_ordered_vocabulary
            from cslr.recognition.dataset import FeatureNormalizer
            from tools.blank_gov.p40_rgb_main import DualInputCTC
        except Exception as exc:                                   # noqa: BLE001
            self.model_error = f"依赖导入失败: {type(exc).__name__}: {exc}"
            return

        try:
            import csv

            from cslr.recognition.gloss_sequence import build_ordered_vocabulary

            # 词表：必须与训练时同参
            labels = {}
            with open(REPO / "data/raw/CE-CSL/label/train.csv",
                      newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    labels[row["Number"]] = row["Gloss"]
            self.voc, _ = build_ordered_vocabulary(
                labels.values(), min_frequency=2, max_tokens=300)

            blob = torch.load(self.checkpoint, map_location="cpu",
                              weights_only=False)
            cfg = blob["config"]
            if int(blob.get("vocab_size", -1)) != int(self.voc.size):
                self.model_error = (
                    "词表大小不一致：checkpoint {} vs 当前 {}".format(
                        blob.get("vocab_size"), self.voc.size))
                return

            self.model = DualInputCTC(
                lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                vocab=int(self.voc.size), hidden=cfg["hidden"],
                layers=cfg["layers"], dropout=cfg["dropout"],
                use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                mode=cfg.get("mode", "add")).to(self.device)
            self.model.load_state_dict(blob["model_state"])
            self.model.eval()
            self.epoch = blob.get("epoch")
            self.train_dev_wer = blob.get("dev_wer")

            # 🔴 不要在这里 fit 归一化器（2026-10-05 移除）。
            # 原代码对推理输入套了 FeatureNormalizer，注释写「与训练一致」，
            # 但**训练侧实际不做归一化**：P40/P42 的 to_batch() 是
            #     lm[i] = torch.from_numpy(s["lm"])
            # 直接吃原始特征。口径不一致的实测代价（P44b，514 条真 dev）：
            #     训练口径（不归一化）WER = 0.5211，hyp<unk> = 33.0%，exact 18/514
            #     服务口径（归一化）  WER = 1.2851，hyp<unk> = 11.2%，exact  0/514
            # 归一化让输入分布偏离模型学过的信号，WER 劣化 2.5 倍、
            # 精确匹配归零、输出长度虚增到 6.63（真实参考均值只有 5.52）。
            # 用户从前端看到「几乎全是 <unk>」正是这个 bug 的直接后果。
            # 保留 self.nrm = None 以兼容旧代码路径的判空。
            self.nrm = None

            from realtime_landmark import RealtimeLandmarkExtractor
            self.extractor = RealtimeLandmarkExtractor()

            self.model_error = None
        except Exception as exc:                                   # noqa: BLE001
            self.model_error = f"模型加载失败: {type(exc).__name__}: {exc}"
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.extractor is not None

    # ------------------------------------------------------------------ 推理
    def predict_video(self, video_path: Path) -> Prediction:
        t0 = time.time()
        if not self.ready:
            return Prediction(
                status="unavailable", label="model_not_loaded",
                gloss_tokens=[], intent="unknown", gloss="", text_zh="",
                confidence=0.0, top_k=[], warnings=[self.model_error or "模型未就绪"],
                latency_ms={"total": 0.0}, model_version=None)

        try:
            t_feat = time.time()
            raw = self.extractor.extract_to_48x368(video_path)
            t_feat_ms = (time.time() - t_feat) * 1000

            # ⚠️ 不做归一化 —— 与训练侧 to_batch() 严格一致（P44b 实测）。
            # 曾在这里套 FeatureNormalizer，注释自称「与训练一致」，实际不一致，
            # 代价是 dev WER 0.5211 -> 1.2851、exact 18 -> 0。
            x = raw
            if self.nrm is not None:
                x = self.nrm.apply(raw)
            x = np.ascontiguousarray(x[None, ...].astype(np.float32))

            t_inf = time.time()
            import torch

            from cslr.recognition.training import decode_batch
            xt = torch.from_numpy(x).to(self.device)
            with torch.no_grad():
                logits = self.model(xt, torch.full((1,), 48, device=self.device,
                                                  dtype=torch.long),
                                    None, None)
                lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            dec, _, _ = decode_batch(lp, np.array([lp.shape[1]]), 1)
            tokens = self.voc.decode(list(dec[0]))
            probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
            t_inf_ms = (time.time() - t_inf) * 1000

            total_ms = (time.time() - t0) * 1000
            if not tokens:
                return Prediction(
                    status="empty", label="no_Gloss", gloss_tokens=[],
                    intent="unknown", gloss="", text_zh="未识别出手语词",
                    confidence=0.0, top_k=[],
                    warnings=["模型未输出任何 gloss —— 特征分布可能与训练不一致"],
                    latency_ms={"feature": t_feat_ms, "inference": t_inf_ms,
                                "total": total_ms},
                    model_version=self._version())

            # 置信度：非 blank 帧上 token 类与 blank 类的平均 margin
            conf = self._confidence(lp[0], dec[0], probs)
            top_k = self._top_k(lp[0], probs, k=5)
            gloss = "/".join(tokens)
            return Prediction(
                status="ok", label=tokens[0], gloss_tokens=list(tokens),
                intent="unknown", gloss=gloss, text_zh=gloss,
                confidence=conf, top_k=top_k,
                warnings=self._warnings(conf),
                latency_ms={"feature": t_feat_ms, "inference": t_inf_ms,
                            "total": total_ms},
                model_version=self._version())

        except Exception as exc:                                   # noqa: BLE001
            total_ms = (time.time() - t0) * 1000
            return Prediction(
                status="error", label="exception", gloss_tokens=[],
                intent="unknown", gloss="", text_zh="",
                confidence=0.0, top_k=[],
                warnings=["{}: {}".format(type(exc).__name__, exc)],
                latency_ms={"total": total_ms}, model_version=None)

    # ------------------------------------------------------------------ 辅助
    def _version(self) -> str:
        return "p42-lm_only-ep{}-devWER{}".format(self.epoch, self.train_dev_wer)

    @staticmethod
    def _confidence(lp_row: np.ndarray, seq: list[int], probs: np.ndarray) -> float:
        """非 blank 帧上，token 类概率与 blank 类概率的平均比值。"""
        blank = np.exp(lp_row[:, 0])
        vals = []
        for t in range(lp_row.shape[0]):
            top = int(np.argmax(lp_row[t]))
            if top != 0:
                vals.append(1.0 - blank[t])
        return round(float(np.mean(vals)), 4) if vals else 0.0

    def _top_k(self, lp_row: np.ndarray, probs: np.ndarray, k: int = 5) -> list[dict]:
        """对每个输出位置取 top-k 候选（去重后按平均概率排序）。"""
        agg: dict[str, list[float]] = {}
        for t in range(lp_row.shape[0]):
            order = np.argsort(-probs[t])[:k]
            for cid in order:
                if cid == 0:
                    continue
                if cid - 1 < self.voc.size:
                    tok = self.voc.tokens[cid - 1]
                    agg.setdefault(tok, []).append(float(probs[t][cid]))
        rows = [{"label": t, "token": t, "intent": None,
                 "confidence": round(float(np.mean(v)), 4)}
                for t, v in agg.items()]
        rows.sort(key=lambda r: -r["confidence"])
        return rows[:5]

    def _warnings(self, conf: float) -> list[str]:
        w = []
        if conf < 0.30:
            w.append("置信度偏低（{:.0%}），识别结果可能不可靠".format(conf))
        return w


def create_ctc_landmark_service() -> CtcLandmarkService:
    """工厂：按 CSLR_CTC_LANDMARK_CKPT 指定加载 P42 checkpoint。"""
    ckpt = Path(os.getenv(
        "CSLR_CTC_LANDMARK_CKPT",
        str(REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt")))
    return CtcLandmarkService(ckpt)
