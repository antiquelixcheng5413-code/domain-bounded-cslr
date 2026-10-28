#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CE-CSL 识别后端服务（可切换模型）。

基类 CslrRecognitionService 封装通用管线：
    上传/录制的视频 -> cv2 抽帧 -> 官方预处理(RGB_FIX+可选 OFFICIAL_PAD) -> 官方 Net -> CTC 贪心解码 -> gloss
精确复刻 tools/blank_gov/p156_test_eval.py 推理口径：
    RGB_FIX=True / BLANK_BIAS=0.0 / img_size=160
    collate 官方补帧 L=ceil(T/4)*4+12（仅 TFNet 发布权重用），首帧重复6 + 末帧补齐
    model(vid, dl, False); LogSoftmax(out[0]); greedy_decode(logp, dl, idx2word)

子类：
    TFNetPublishedService ：发布 TFNet(resnet34MAM)@160 RGB 权重，免训练部署（dev 36.81 / test 35.66）
    VacTrainedService     ：我们自己 from-scratch 训练的最佳 VAC(resnet18 2D)@160（P150，dev 53.73）

资源缺失时返回 model_unavailable（不伪造预测），与现有服务一致。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "external/TFNet"

# 单条视频最多抽多少帧（CE-CSL 训练用 193；更大只均匀下采样，避免 CPU 推理爆内存）
MAX_FRAMES = 256
RESIZE_HW = 256  # 官方 P77 已把磁盘帧 resize 到 ~256，这里对齐后再 CenterCrop(160)


class CslrRecognitionService:
    """视频 -> 官方 Net -> CTC 贪心 -> CE-CSL gloss。可被子类参数化。"""

    # ---- 子类可覆盖的默认（下面是 TFNet 发布权重口径） ----
    DEFAULT_WEIGHTS = Path(
        "/mnt/c/Users/su127/Desktop/中文手语识别/官方权重/"
        "TFNet-CE-CSL-CSLDaily-32.46.pth"
    )
    ENV_WEIGHTS = "CSLR_TFNET_WEIGHTS"
    MODEL_MODULE = "TFNet"          # Net.moduleNet 的 module 参数
    CK_KEY = "moduleNet_state_dict" # 权重文件里的状态字典键
    STRICT = True                  # load_state_dict 严格度
    OFFICIAL_PAD = True            # 是否用官方补帧（TFNet 用；VAC 训练用 False）
    MODEL_VERSION = "tfnet"
    LABEL = "TFNet（发布权重）"
    WER = "dev 36.81% / test 35.66%"
    WARN = "发布权重免训练部署（CE-CSL, @160 RGB）。输出为 gloss 序列，未做中文翻译。"

    def __init__(self, weights: str | None = None, img_size: int = 160,
                 device: str | None = None) -> None:
        self.weights = Path(
            weights or os.getenv(self.ENV_WEIGHTS, str(self.DEFAULT_WEIGHTS))
        )
        self.img_size = int(os.getenv("CSLR_TFNET_IMG", str(img_size)))
        self.device = self._auto_device(device)
        self.model_error = self._check_resources()
        # 惰性加载
        self._model = None
        self._idx2word = None
        self._P = None
        self._torch = None
        self.model_version = self.MODEL_VERSION

    # ---- 设备 ------------------------------------------------------------
    def _auto_device(self, device: str | None) -> str:
        if device is None:
            device = os.getenv("CSLR_TFNET_DEVICE", "auto")
        if device and device != "auto":
            return device
        import torch  # noqa: F401
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 资源检查 --------------------------------------------------------
    def _check_resources(self) -> str | None:
        if not self.weights.exists():
            return "missing weights: " + str(self.weights)
        if not (EXT / "data/CE-CSL/train.csv").exists():
            return "missing CE-CSL vocab csv under external/TFNet/data/CE-CSL"
        return None

    # ---- 加载（惰性） ----------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401
        tb = str(REPO / "tools" / "blank_gov")
        for p in (tb, str(REPO / "src"), str(EXT)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import p78_train_official_tfnet as P  # type: ignore
        self._P = P
        P.RGB_FIX = True
        P.OFFICIAL_PAD = self.OFFICIAL_PAD
        P.BLANK_BIAS = 0.0

        word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
            str(P.CSV_DIR / "train.csv"),
            str(P.CSV_DIR / "dev.csv"),
            str(P.CSV_DIR / "test.csv"),
            "CE-CSL",
        )
        dev = torch.device(self.device)
        model = P.Net.moduleNet(
            1024, wordSetNum + 1, self.MODEL_MODULE, dev, "CE-CSL", True
        ).to(dev)
        ck = torch.load(
            str(self.weights), map_location="cpu", weights_only=False, mmap=True
        )
        sd = ck[self.CK_KEY] if self.CK_KEY in ck else ck.get("model_state", ck)
        model.load_state_dict(sd, strict=self.STRICT)
        model.eval()
        self._model = model
        self._idx2word = idx2word
        self._torch = torch

    # ---- 抽帧 + 官方预处理 ----------------------------------------------
    def _load_frames(self, video_path: Path) -> "np.ndarray":
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
        raw = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None and frame.ndim == 3:
                raw.append(frame)  # BGR uint8，匹配 RGB_FIX=True（不 cvtColor）
        cap.release()
        if not raw:
            raise ValueError("视频中未读取到任何帧")

        if len(raw) > MAX_FRAMES:
            idxs = np.linspace(0, len(raw) - 1, MAX_FRAMES).astype(int)
            raw = [raw[i] for i in idxs]

        imgs = [cv2.resize(f, (RESIZE_HW, RESIZE_HW)) for f in raw]
        tf = self._P.build_transforms(hflip=False, img_size=self.img_size)[1]
        seq = tf(imgs)                      # [T,3,H,W] float [0,1]
        seq = seq.float() / 127.5 - 1.0     # -> [-1,1]
        return seq.numpy() if hasattr(seq, "numpy") else np.asarray(seq)

    # ---- 解码（带置信度，镜像 greedy_decode） ----------------------------
    def _decode(self, logp, L):
        seq = logp.detach().cpu()
        if seq.shape[0] != 1 and seq.shape[1] == 1:
            seq = seq.transpose(0, 1)       # -> [B,T,C]
        t = min(self._P._conv_len(int(L)), seq.shape[1])
        best = seq[0, :t].argmax(-1).numpy()
        confs = seq[0, :t].max(-1).values.numpy()
        toks, mc, prev = [], [], -1
        for ti in range(t):
            k = int(best[ti])
            if k != prev and k != 0 and 0 <= k < len(self._idx2word):
                toks.append(self._idx2word[k])
                mc.append(float(confs[ti]))
            prev = k
        conf = float(np.exp(np.mean(mc))) if mc else 0.0
        return toks, conf

    # ---- 主入口 ----------------------------------------------------------
    def predict_video(self, video_path: Path) -> dict:
        started = time.perf_counter()
        latency: dict[str, float] = {}

        if self.model_error is not None:
            return self._unavailable(latency, started)

        try:
            self._ensure_loaded()
            torch = self._torch
            dev = torch.device(self.device)

            t0 = time.perf_counter()
            seq = torch.from_numpy(self._load_frames(video_path)).to(dev)
            latency["extract_frames"] = round((time.perf_counter() - t0) * 1000, 2)

            T = int(seq.shape[0])
            if self.OFFICIAL_PAD:
                L = ((T + 3) // 4) * 4 + 12
                padded = torch.cat(
                    [seq[0][None].expand(6, -1, -1, -1), seq,
                     seq[-1][None].expand(L - 6 - T, -1, -1, -1)],
                    dim=0,
                )
            else:
                L = T
                padded = seq
            vid = padded.unsqueeze(0)        # [1, L, 3, H, W] 或 [1, T, 3, H, W]
            dl = torch.LongTensor([[L]])

            t1 = time.perf_counter()
            with torch.no_grad():
                out = self._model(vid.to(dev), dl, False)
                logp = torch.nn.functional.log_softmax(out[0].float(), dim=-1)
                toks, conf = self._decode(logp, L)
            latency["infer"] = round((time.perf_counter() - t1) * 1000, 2)
            latency["total"] = round((time.perf_counter() - started) * 1000, 2)

            gloss = "/".join(toks) if toks else "UNKNOWN"
            text_zh = gloss if toks else "（未能识别出手语序列）"
            return {
                "status": "ok",
                "label": gloss,
                "gloss_tokens": toks,
                "intent": gloss,
                "gloss": gloss,
                "text_zh": text_zh,
                "confidence": conf,
                "top_k": [],
                "warnings": [self.WARN],
                "latency_ms": latency,
                "model_version": self.model_version,
            }
        except Exception as exc:  # noqa: BLE001
            latency["total"] = round((time.perf_counter() - started) * 1000, 2)
            return {
                "status": "error",
                "label": "unknown",
                "gloss_tokens": [],
                "intent": "unknown",
                "gloss": "UNKNOWN",
                "text_zh": f"识别失败：{exc}",
                "confidence": 0.0,
                "top_k": [],
                "warnings": [str(exc)],
                "latency_ms": latency,
                "model_version": self.model_version,
            }

    def _unavailable(self, latency: dict, started: float) -> dict:
        latency["total"] = round((time.perf_counter() - started) * 1000, 2)
        return {
            "status": "model_unavailable",
            "label": "unknown",
            "gloss_tokens": [],
            "intent": "unknown",
            "gloss": "UNKNOWN",
            "text_zh": "模型权重未就位，当前不能进行真实识别。",
            "confidence": 0.0,
            "top_k": [],
            "warnings": [self.model_error or "未知资源缺失"],
            "latency_ms": latency,
            "model_version": self.model_version,
        }

    @property
    def ready(self) -> bool:
        return self.model_error is None

    @property
    def demo_mode(self) -> bool:
        return False

    @property
    def info(self) -> dict:
        return {
            "name": self.model_version,
            "label": self.LABEL,
            "wer": self.WER,
            "ready": self.ready,
        }


class TFNetPublishedService(CslrRecognitionService):
    """发布 TFNet(resnet34MAM)@160 RGB 权重，免训练部署。"""
    pass


class VacTrainedService(CslrRecognitionService):
    """我们自己 from-scratch 训练的最佳 VAC(resnet18 2D)@160（P150）。"""
    DEFAULT_WEIGHTS = REPO / "artifacts/checkpoints/p150-rgbfix-best.pt"
    ENV_WEIGHTS = "CSLR_VAC_WEIGHTS"
    MODEL_MODULE = "VAC"
    CK_KEY = "model_state"
    STRICT = False                 # VAC 主干不含辅助头键时用宽松加载，推理只看 CTC(out[0])
    OFFICIAL_PAD = False           # P150 训练用默认 collate（真实帧数，不补帧）
    MODEL_VERSION = "vac"
    LABEL = "VAC（自训 P150）"
    WER = "dev 53.73%"
    WARN = "我们 from-scratch 训练的最佳 VAC@160（P150, dev 53.73%），输出为 gloss 序列。"


def create_tfnet_service() -> TFNetPublishedService:
    return TFNetPublishedService()


def create_vac_service() -> VacTrainedService:
    return VacTrainedService()
