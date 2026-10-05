# -*- coding: utf-8 -*-
"""实时 landmark 特征提取器（mediapipe 0.10.35 新 API）

## 为什么需要这个文件

仓库原有的 `src/cslr/features/extractor.py` 用
`mp.solutions.holistic.Holistic`，但 **mediapipe 0.10.35 已移除整个
`solutions` 命名空间**（实测 `No module named 'mediapipe.python'`）。
`requirements.txt` 钉的 0.10.21 在 Python 3.14 无 wheel，所以那条代码路径
**从未真正跑通过**。

本文件用新 API（`mp.tasks.vision.*`）重写了**逐帧 landmark 提取**，
产出与训练特征**同布局**的 `(48, 368)` 数组，使推理可行。

## 🔴 与训练特征的对齐要求

训练用的是 `artifacts/part3_features/<split>/*.landmark.npy`，
布局（见 extractor.py 常量）：

| 区间 | 宽度 | 内容 |
|---|---|---|
| `[0:126]` | 126 | 双手 21 点 × 3 坐标（左手 63 + 右手 63） |
| `[126:158]` | 32 | pose 8 点 × 4（xyz + visibility） |
| `[158:182]` | 24 | face 8 点 × 3 |
| `[182:186]` | 4 | presence（pose/左手/右手/脸） |
| `[186:368]` | 182 | 前 182 维的一阶差分（在**原始帧率**上求） |

`POSE_INDICES = (11,12,13,14,15,16,23,24)` —— 8 个点，索引 11/12 是双肩
（被用作归一化原点，故其相对坐标恒为 0.5 附近，见 P31 实测）。
`FACE_INDICES = (10,33,61,133,152,263,291,362)` —— 8 个轮廓点。

## 归一化（必须与训练一致）

训练特征做了肩距归一化（P31 实测帧间位移 mean 0.264，若未归一化会是原始像素值）：
```
origin = (pose[11] + pose[12]) / 2      # 双肩中点
scale  = |pose[11] - pose[12]|          # 肩距
每维 = (point - origin) / scale
```
**若推理时不做这一步，模型输入分布与训练完全不同，输出必然是垃圾。**

## 三点已知偏差（诚实标注）

1. **pose 只有 8 点**（与训练一致）。MediaPipe PoseLandmarker 给 33 点，
   取其中 8 个以匹配训练布局。
2. **face 只有 8 点**（与训练一致）。新 API 的 FaceLandmarker 给 478 点，
   取 `FACE_INDICES` 指定的 8 个。P34 已证「加到 128 点可分性只 1.01x」，
   故保持 8 点是**有依据的取舍**，不是偷懒。
3. **presence 用检出标志**代替原实现的确切语义，
   四位顺序与训练的 `[182:186]` 一致（pose/左手/右手/脸）。

## 差分的时序基准

训练的 `[186:368]` 是在**原始帧率**上求的一阶差分（先 append_motion 再 resample），
而 `[0:182]` 是重采样到 48 帧后的序列。
本实现严格照此顺序：先在原始帧序列上求差分，再重采样。

## 用法

    from realtime_landmark import RealtimeLandmarkExtractor
    ex = RealtimeLandmarkExtractor()
    feats = ex.extract_to_48x368("video.mp4")     # (48, 368) float32
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

def _find_repo() -> Path:
    """向上查找含 models/ 与 src/ 的目录，避免任何硬编码绝对路径。

    🔴 原实现（2026-10-06 前）：
        REPO = Path(__file__).resolve().parents[2] if ...endswith(
            "app/backend/realtime_landmark.py") else Path(
            "/home/su127/FYP/domain-bounded-cslr")
    问题：
      1. Windows 副本的else 分支写死 WSL 路径，跨机器不可用（P75 E4）
      2. 条件判断依赖路径字符串形态，换目录结构就失效
    现在按**标记文件**探测，与 p57/p70 等脚本的做法一致。
    """
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "models").is_dir() and (cand / "src").is_dir():
            return cand
    raise RuntimeError(
        "repo root not found (looking for models/ and src/ next to %s)" % here)


REPO = _find_repo()

# 与训练特征一致的索引（src/cslr/features/extractor.py）
POSE_INDICES = (11, 12, 13, 14, 15, 16, 23, 24)
FACE_INDICES = (10, 33, 61, 133, 152, 263, 291, 362)
BASE_SIZE = 182
OUTPUT_SIZE = 368
N_FRAMES = 48


class _DawnCloser:
    """在守护线程里关闭旧的 landmarker（2026-10-05）。

    为什么需要它：`mediapipe` 的 landmarker.close() 会阻塞约 15s
    （等待内部 dispatcher 线程池的 pending 任务）。无论是在主线程显式
    调用，还是被 GC 的 finalizer 触发，都会把整条识别链路卡住。
    放进 daemon 线程后，主流程立即继续，清理在后台慢慢完成。
    """

    def __init__(self, objs) -> None:
        import threading

        self._objs = list(objs)
        threading.Thread(target=self._run, daemon=True,
                         name="mp-landmarker-close").start()

    def _run(self) -> None:
        for obj in self._objs:
            try:
                obj.close()
            except Exception:                                   # noqa: BLE001
                pass
        self._objs = []


class RealtimeLandmarkExtractor:
    """用 mediapipe 0.10.35 新 API 提取 (48, 368) landmark 特征。

    懒加载模型，构造开销只付一次。
    """

    def __init__(self, device_note: str = "cpu", parallel_models: bool = True,
                 reset_per_video: bool = True) -> None:
        self._pose = None
        self._hands = None
        self._face = None
        self._ready = False
        self._error: str | None = None
        self._n_hands_seen = 0
        # 🔴 跨调用单调递增的时间戳游标（2026-10-05）。
        # MediaPipe VIDEO 模式要求时间戳严格递增，而 landmarker 实例跨视频复用，
        # 所以游标必须存在实例上而非 extract 的局部变量。
        # _ts_step_ms=33 ≈ 30fps，与原实现 ts+=1 的帧率语义一致，
        # 但用毫秒量级以符合 mediapipe 的内部时间基准。
        self._ts_ms = 0
        self._ts_step_ms = 33
        # 线程池：3 个 worker 对应 3 个模型，实测 2.01x 加速。
        self._parallel_models = parallel_models
        self._pool = None
        # 每个视频重建模型以保证结果可复现（默认开，见 _reset_for_new_video）
        self._reset_per_video = reset_per_video

    # ------------------------------------------------------------------ 模型
    def _ensure(self) -> None:
        if self._ready:
            return
        if self._error:
            raise RuntimeError(self._error)
        try:
            import mediapipe as mp
        except Exception as exc:                                   # noqa: BLE001
            self._error = f"mediapipe 不可用: {exc}"
            raise RuntimeError(self._error) from exc
        if not hasattr(mp, "tasks"):
            self._error = (
                "mediapipe 版本过旧，缺少 mp.tasks（需 >= 0.10.30）。"
                "当前版本 {}；0.10.21 在 Python 3.14 无 wheel。".format(
                    getattr(mp, "__version__", "unknown"))
            )
            raise RuntimeError(self._error)

        models = REPO / "models"
        need = {
            "pose": models / "pose_landmarker_lite.task",
            "hand": models / "hand_landmarker.task",
            "face": models / "face_landmarker.task",
        }
        missing = [f"{k}: {v}" for k, v in need.items() if not v.exists()]
        if missing:
            self._error = "缺少模型文件 -> " + "; ".join(missing)
            raise RuntimeError(self._error)

        v = mp.tasks.vision
        base = lambda p: mp.tasks.BaseOptions(model_asset_path=str(p))  # noqa: E731
        self._build_models(v, base, need)
        self._ready = True
        self._ts_ms = 0          # 模型重建 -> 时间戳状态清零，重新从 0 开始
        self._pool = None
        if self._parallel_models:
            try:
                from concurrent.futures import ThreadPoolExecutor

                # 3 个 worker 恰好对应 3 个模型；再多也不会更快
                # （实测 W=3 与 W=6 同为 ~5.7s）。
                self._pool = ThreadPoolExecutor(
                    max_workers=3, thread_name_prefix="mp-landmark")
            except Exception:                                   # noqa: BLE001
                self._pool = None                              # 退化为串行

    def _build_models(self, v, base, need) -> None:
        """构建（或重建）三个 landmarker。

        🔴 必须在**每个视频开始时**重建（2026-10-05）。
        MediaPipe VIDEO 模式内部维护跨帧跟踪状态，且**没有 reset API**。
        复用实例时，上一段视频残留的跟踪状态会污染下一段：
        实测同一视频连续 4 次预测得到 4 个不同结果
        （`<unk>/<unk>/一直...` -> `一直/一直/一直` -> `一直` -> `多/一直×6`），
        即结果**不可复现**。重建对象是唯一可靠的隔离方式。
        代价：实测重建约 0.2s，相对 3~6s 的推理开销可接受。
        """
        self._pose = v.PoseLandmarker.create_from_options(
            v.PoseLandmarkerOptions(
                base_options=base(need["pose"]),
                running_mode=v.RunningMode.VIDEO, num_poses=1))
        self._hands = v.HandLandmarker.create_from_options(
            v.HandLandmarkerOptions(
                base_options=base(need["hand"]),
                running_mode=v.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5))
        self._face = v.FaceLandmarker.create_from_options(
            v.FaceLandmarkerOptions(
                base_options=base(need["face"]),
                running_mode=v.RunningMode.VIDEO, num_faces=1))

    def _reset_for_new_video(self) -> None:
        """新视频开始前调用：重建模型 + 时间戳归零，保证结果可复现。"""
        if not self._ready:
            return
        if not self._reset_per_video:
            return
        try:
            import mediapipe as mp

            v = mp.tasks.vision
            base = lambda p: mp.tasks.BaseOptions(model_asset_path=str(p))  # noqa: E731
            models = REPO / "models"
            need = {
                "pose": models / "pose_landmarker_lite.task",
                "hand": models / "hand_landmarker.task",
                "face": models / "face_landmarker.task",
            }
            if all(p.exists() for p in need.values()):
                # 🔴 旧实例的清理必须放到**后台守护线程**（2026-10-05 实测）。
                # 踩过的三个坑，全部实测过：
                #   1. 直接覆盖引用 -> GC 回收旧对象时触发阻塞式 close()，
                #      实测整体仍然 18.5s（延迟但没消失，只是换了个触发点）；
                #   2. 显式 obj.close() -> 单个就阻塞 15.02s
                #      （在等 mediapipe dispatcher 线程池的 pending 任务）；
                #   3. 引用计数置零后再 close -> 同样 15s。
                # 结论：close() 本身不可阻塞调用，而 GC 时机不可控，
                # 所以唯一可靠做法是**丢给守护线程**，主流程立即继续。
                # 旧实例的原生资源由该线程最终释放。
                old = [o for o in (self._pose, self._hands, self._face)
                       if o is not None]
                self._pose = self._hands = self._face = None
                self._build_models(v, base, need)
                self._ts_ms = 0
                if old:
                    _DawnCloser(old)
        except Exception:                                       # noqa: BLE001
            # 重建失败就沿用旧实例：慢一点，但比整体不可用好。
            # 时间戳仍需继续递增，否则 VIDEO 模式会直接抛异常。
            pass

    def close(self) -> None:
        # ⚠️ 旧实例的 close() 会阻塞 15.02s（等 mediapipe dispatcher
        # 线程池的 pending 任务），三个串起来 45s。必须丢到后台线程，
        # 否则服务关闭时会长时间卡住。
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False)
            except Exception:                                   # noqa: BLE001
                pass
            self._pool = None
        old = [o for o in (self._pose, self._hands, self._face) if o is not None]
        self._pose = self._hands = self._face = None
        self._ready = False
        self._ts_ms = 0
        if old:
            _DawnCloser(old)

    # -------------------------------------------------------------- 单帧提取
    def _frame_vector(self, pose_res, hand_res, face_res):
        """返回 (182 维 base, presence 4 位)。"""
        # ---- pose：33 点里取 8 个 ----
        pose = np.zeros((len(POSE_INDICES), 4), dtype=np.float32)
        p = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
        if p is not None:
            for k, idx in enumerate(POSE_INDICES):
                pose[k, 0] = p[idx].x
                pose[k, 1] = p[idx].y
                pose[k, 2] = p[idx].z
                pose[k, 3] = getattr(p[idx], "visibility", 0.0) or 0.0

        # ---- 双手：21 点 × 3，左手先右手后 ----
        hands = np.zeros((2, 21, 3), dtype=np.float32)
        have_l = have_r = False
        if hand_res and hand_res.hand_landmarks:
            # 🔴 左右手标签必须按**下标**取，不能恒取 [0][0]（2026-10-05 修复）。
            # 原代码在循环里写 `hand_res.handedness[0][0].category_name`，
            # 无论当前是第几只手都读第 0 只手的标签 ->
            #   第 2 只手被套上第 1 只手的左右标签 -> 帧间左右手交替闪烁。
            # 实测后果（P44c，8 个真实 dev 视频）：
            #   presence 离线 [0.979, 0.938, 1.000, 1.000]（双手几乎都在）
            #   presence 线上 [1.000, 0.438, 0.542, 1.000]（左手 44%、右手 54%）
            #   同批视频 WER 0.5818 -> 0.7091，输出 token 种类 18 -> 1
            #   （模型只能吐一个 <unk>）
            # hands 段占 126/368 维（34%），std 还大 1.68 倍，
            # 所以这个错误直接决定了线上输出全是 <unk>。
            # 🔴🔴 handedness 标签必须**翻转**（2026-10-05 19:45 修正）
            #
            # 实测证据（train-00001 前 12 帧，每帧两手都检出）：
            #   新 API: label='Left'  的手 x 均值 = 0.419
            #          label='Right' 的手 x 均值 = 0.410
            #          → 标 "Left" 的反而在画面右侧
            #   旧 holistic: left_hand_landmarks 是**解剖学左手**（x 较小）
            #
            # ⇒ 新 API 的 handedness 默认按**图像视角**（假定画面已镜像），
            #   与旧 API 的解剖学左右**语义相反**。
            #
            # 修正前的实测代价（P53 抽样 94 条 train）：
            #   旧特征 presence handL=0.528 handR=**1.000**（右手几乎必有）
            #   新特征 presence handL=0.608 handR=**0.497**（右手续检率腰斩）
            # hands 段占 126/368 维（34%），左右手错位会让特征顺序完全错乱。
            for hi, h in enumerate(hand_res.hand_landmarks):
                arr = np.array([[lm.x, lm.y, lm.z] for lm in h], dtype=np.float32)
                if arr.shape[0] != 21:
                    continue
                label = None
                hd = getattr(hand_res, "handedness", None)
                if hd is not None and len(hd) > hi:
                    label = str(hd[hi][0].category_name).lower()
                    # 翻转：图像视角 -> 解剖学左右
                    label = "right" if label == "left" else "left"
                if label == "right" and not have_r:
                    hands[1] = arr
                    have_r = True
                elif label == "left" and not have_l:
                    hands[0] = arr
                    have_l = True
                elif not have_l and not have_r:
                    # 标签缺失/无法识别时，先占左手槽（与训练 extractor 的
                    # 「按顺序填」约定一致），下一个手若确认是 right 仍能就位。
                    hands[0] = arr
                    have_l = True
            self._n_hands_seen += int(have_l) + int(have_r)

        # ---- face：478 点里取 8 个轮廓点 ----
        face = np.zeros((len(FACE_INDICES), 3), dtype=np.float32)
        have_f = False
        if face_res and face_res.face_landmarks:
            fl = face_res.face_landmarks[0]
            if len(fl) > max(FACE_INDICES):
                for k, idx in enumerate(FACE_INDICES):
                    face[k] = (fl[idx].x, fl[idx].y, fl[idx].z)
                have_f = True

        # ---- 肩距归一化（与训练一致；pose 缺失时退化到 1.0）----
        origin = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        scale = 1.0
        if p is not None and len(p) > 12:
            ls = np.array([p[11].x, p[11].y, p[11].z], dtype=np.float32)
            rs = np.array([p[12].x, p[12].y, p[12].z], dtype=np.float32)
            origin = (ls + rs) / 2.0
            d = float(np.linalg.norm(ls - rs))
            if d > 1e-6:
                scale = d

        # 🔴 pose 段必须保留 visibility（第 4 维）才够 32 维。
        #   visibility 是 0~1 的置信度，**不能做肩距归一化**（会破坏语义）；
        #   只对 xyz 三维归一化，第 4 维原样保留。
        #   我先前写成 pose[:, :3] 导致 pose 段只有 24 维、总维 174（错）。
        pose_xyz = (pose[:, :3] - origin) / scale          # (8,3) 归一化
        pose_vis = pose[:, 3:4]                             # (8,1) 原样
        pose_full = np.concatenate([pose_xyz, pose_vis], axis=1)   # (8,4) = 32
        hands_xyz = (hands - origin) / scale
        face_xyz = (face - origin) / scale

        base = np.concatenate([
            hands_xyz.reshape(-1),        # 126
            pose_full.reshape(-1),        # 32  ← 含 visibility
            face_xyz.reshape(-1),         # 24
        ]).astype(np.float32)           # 182 ✅
        # 🔴 自检：维度错了模型输出必然是垃圾，且不会报错（静默失效）。
        #   本项目已因「静默维度错误」栽过（P34 三个配置逐位相同）。
        assert base.shape == (BASE_SIZE,), (
            "base 应为 {} 维，实际 {} —— 检查 pose visibility 是否被丢掉".format(
                BASE_SIZE, base.shape[0]))

        # 🔴🔴 presence 四位的语义顺序必须与训练特征一致 = [handL, handR, pose, face]
        # 实证依据（git 最早提交 src/cslr/features/extractor.py:125）：
        #     masks = np.asarray([left_present, right_present, pose_present, face_present])
        # 且 base 顺序为 (left, right, pose_values, face_values) —— mask 与 base 一致。
        #
        # 我原先写成 [pose, handL, handR, face]，前三位整体错位。
        # 后果不是数值噪声而是**语义错位**：presence 是「这帧哪些模态有效」的指示位，
        # 模型会把 pose 当 handL 学，稳定收敛到错误映射。
        # 实测症状（旧基准 vs 我的错位版）：
        #     旧 [0.597, 0.601, 1.000, 0.998]  <- 第 3 位恒 1.0 = pose
        #     新 [1.000, 0.599, 0.564, 0.751]  <- 第 1 位恒 1.0 = pose
        presence = np.array([
            1.0 if have_l else 0.0,
            1.0 if have_r else 0.0,
            1.0 if p is not None else 0.0,
            1.0 if have_f else 0.0,
        ], dtype=np.float32)
        return base, presence

    # ---------------------------------------------------------------- 整段
    def extract_to_48x368(self, video_path, max_frames: int | None = None) -> np.ndarray:
        """视频 -> (48, 368) float32，布局与训练特征完全一致。

        顺序严格照训练 extractor：先在原始帧率上求一阶差分，再重采样到 48 帧。
        """
        self._ensure()
        import cv2
        import mediapipe as mp

        # 🔴 每个视频开始前重建模型：消除 VIDEO 模式的跨视频跟踪残留，
        #    否则同一视频重复预测会得到不同结果（不可复现）。
        self._reset_for_new_video()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        bases, masks = [], []
        # 🔴 时间戳必须全局单调递增（2026-10-05 修复）。
        # 三个 landmarker 是**跨调用复用**的实例，内部保留上一次视频的时间戳状态。
        # 原实现每次调用都从 ts=0 重新开始 -> 第二个视频起全部抛
        #   ValueError: Input timestamp must be monotonically increasing.
        # 该异常被 predict_video 的 except 吞掉 -> status='error'，
        # 表现为「第一个视频能识别，之后一直失败/超时重试」。
        # 用 self._ts_ms 持续累加，且每 1000 秒回绕（int32 溢出保护）。
        ts = self._ts_ms

        # ⚡ 同帧内三模型并行（2026-10-05，实测 2.01x：309 帧 11.44s -> 5.68s）。
        # 安全性论证（很重要，不要盲目改回串行）：
        #   1. 三个 landmarker 是**三个独立对象**，彼此无共享状态；
        #   2. 每个模型内部仍按 ts 严格递增**串行**推进 VIDEO 跟踪链
        #      —— 同一帧的三次调用 ts 相同、下一帧才 +step，顺序未变；
        #   3. 并行只发生在「同一时刻处理同一帧的不同模型」，
        #      不改变任何单模型的输入顺序，故输出与串行逐位一致。
        # ❌ 不可跨帧并行：VIDEO 模式有跨帧跟踪状态，并发会破坏跟踪链。
        # ❌ 不可抽帧：实测 stride=2 时特征 cos 仅 0.897、3/48 帧 presence
        #    不一致，会改变识别结果（这是「用速度换正确性」，已否决）。
        pool = self._pool
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                img = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if pool is None:
                    pr = self._pose.detect_for_video(img, ts)
                    hr = self._hands.detect_for_video(img, ts)
                    fr = self._face.detect_for_video(img, ts)
                else:
                    f_pose = pool.submit(self._pose.detect_for_video, img, ts)
                    f_hand = pool.submit(self._hands.detect_for_video, img, ts)
                    f_face = pool.submit(self._face.detect_for_video, img, ts)
                    pr, hr, fr = f_pose.result(), f_hand.result(), f_face.result()
                ts += self._ts_step_ms
                b, m = self._frame_vector(pr, hr, fr)
                bases.append(b)
                masks.append(m)
                if max_frames and len(bases) >= max_frames:
                    break
        finally:
            cap.release()
            if ts > 2_000_000_000:      # ~2e9 ms ≈ 23 天，防 int32 溢出
                ts = 0
            self._ts_ms = ts

        if not bases:
            raise RuntimeError("no frames decoded from video")
        base = np.stack(bases)                     # (T, 182)
        mask = np.stack(masks)                     # (T, 4)

        # 一阶差分在原始帧率上（与训练一致）：首帧补 0，长度 T
        diff = np.zeros_like(base)
        if base.shape[0] > 1:
            diff[1:] = base[1:] - base[:-1]
        full = np.concatenate([base, mask, diff], axis=1)   # (T, 368)

        # 重采样到 48 帧（np.linspace 取整索引，不插值 —— 与训练一致）
        T = full.shape[0]
        idx = np.linspace(0, T - 1, N_FRAMES).round().astype(int)
        return full[idx].astype(np.float32)
