# 官方 TFNet 代码核对报告

**官方仓库**：`https://github.com/woshisad159/TFNet`（作者 `woshisad159` = CE-CSL 论文作者 Zhu Qidan）
**本地位置**：`external/TFNet/`（`git clone --depth 1`）
**论文**：arXiv **2409.11960**《A Chinese Continuous Sign Language Dataset Based on Complex Environments》
**核对时间**：2026-10-06

> 本文所有结论均来自**逐行读本地 clone 的官方源码**，不是从论文转述。
> 之前的教训：我曾照论文描述自己写 `official_tfnet.py`，错了 4 处（见 §6）。

---

## 1. 官方 TFNet 完整结构（`Net.py` → `moduleChoice=="TFNet"`）

```
输入 [B, T, C, H, W]  ← RGB 帧序列（256×256，训练时 RandomCrop→224）
  │
  ├─ Module.resnet34MAM()          帧级提取器 F_f，fc=Identity → 512 维/帧
  │     返回 (x, outData1, outData2, outData3)   ← 4 个值，TFNet 分支只用 x
  │
  ├─ framewise [B, T, 512]
  │     ├─ 时域分支：TemporalConv(512→h, conv_type=2) → BiLSTM(2 层双向) → classifier11 / classifier22
  │     └─ 频域分支：fft(framewise, dim=-1) → abs() → TemporalConv → BiLSTM → classifier33 / classifier44
  │
  └─ 融合：x2 = outputs['predictions'] + outputs1['predictions'] → classifier55
     推理时（isTrain=False）：logProbs1 = logProbs5（即用融合头）
```

**分类头共享关系**：`classifier11 is classifier22`、`classifier33 is classifier34`（实为 33/44 共享）、
`classifier55` 独立。共 5 个 logProbs 返回。

---

## 2. ⭐ 主干是 3D ResNet34-MAM，不是 2D（`Module.py`）

```python
class ResNet34MAM:                    # 自定义 BasicBlock / _make_layer
    # 主卷积是 nn.Conv3d
    def forward(self, x):
        return x, outData1, outData2, outData3

def resnet34MAM(**kwargs):
    model = ResNet34MAM(BasicBlock, [3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet34'])   # ImageNet 权重
    checkpoint[ln] = checkpoint[ln].unsqueeze(2)               # 2D → 3D 膨胀
```

**⇒ 官方 = ImageNet 预训练的 2D 权重膨胀成 3D 卷积 + 逐层 MotorAttention。**

**这解释了 landmark 为什么打不过**：landmark 把空间压成 21×3 个点，
丢掉的正是 ResNet 的局部视觉纹理/形状先验。而 TFNet 的设计动机就是
「复杂背景下手部易糊、易被遮挡」—— 官方用 RGB 纹理补，我们没有这个信息源。

`MotorAttention`（MAM）的内部实现见 `Module.py`，本次未逐行细读，
**后续必须读**（不要再照转述写）。

---

## 3. ⭐ TemporalConv 真实结构（我写错了）

```python
# conv_type == 2
self.kernel_size = ['K5', "P2", 'K5', "P2"]
```

```
Conv1d(input_size → hidden, kernel_size=5, stride=1, padding=0)
BatchNorm1d → ReLU → MaxPool1d(kernel_size=2, ceil_mode=False)
Conv1d(hidden → hidden, kernel_size=5, stride=1, padding=0)
BatchNorm1d → ReLU → MaxPool1d(kernel_size=2, ceil_mode=False)
```

**长度变化**：`padding=0` ⇒ 每次卷积减 4 帧。
```
L → L-4 → floor((L-4)/2) → floor((L-4)/2)-4 → /2
```
**⇒ 48 帧 → 22 → 9，只剩 9 个时间步。**（官方靠 `collate_fn` 的
`left_pad=6` + `total_stride=4` 的补帧保证长度整齐，见 §5）

---

## 4. ⭐ NormLinear 不是 nn.Linear

```python
class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
    def forward(self, x):
        return torch.matmul(x, F.normalize(self.weight, dim=0))   # 按输出通道 L2 归一化
```

无 bias，权重按**输出通道**做 L2 归一化。官方用它替代所有输出层，
**推测动机是防输出层范数失衡**（未在论文中说明，属代码观察）。

---

## 5. ⭐ 数据管线（我们已按此实现 RGB 提取）

### 5.1 预处理（`CE-CSLDataPreProcess.py`）
```python
vid = imageio.get_reader(videoPath)
for i in range(vid.count_frames()):
    image = cv2.cvtColor(vid.get_data(i), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    cv2.imencode('.jpg', image)[1].tofile(f"{sample_id}/{i:05d}.jpg")
```

### 5.2 Dataset（`DataProcessMoudle.py::MyDataset`，CE-CSL 分支）
```python
ImageSeq = sorted(os.listdir(fn))          # fn = 帧目录
indices = self.sample_indices(len(ImageSeq))
frames = [os.path.join(fn, i) for i in ImageSeq]
frames = [frames[i] for i in indices]
imgSeq = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (256,256)) for p in frames]
imgSeq = self.transform(imgSeq)
imgSeq = imgSeq.float() / 127.5 - 1         # 归一化到 [-1, 1]
```
`sample_indices(n)` = `np.linspace(0, n-1, num=int(n//1))` = **不抽帧，全用**。

### 5.3 增强（`Train.py`）
```python
transform     = Compose([RandomCrop(224), RandomHorizontalFlip(0.5), ToTensor(), TemporalRescale(0.2)])
transformTest = Compose([CenterCrop(224), ToTensor()])
```

| 增强 | 说明 | 我们的处理 |
|---|---|---|
| `RandomCrop(224)` | 从 256×256 随机裁到 224×224 | 必须实现 |
| `RandomHorizontalFlip(0.5)` | **左右镜像翻转** | ⚠️ **对中文手语是镜像手势，语义可能相反。官方就这么用，必须实测开关对比** |
| `TemporalRescale(0.2)` | 时间缩放 ±20% | 相当于我 P72 的 jitter（方向一致） |
| 归一化 | `x/127.5 - 1` → [-1,1] | **不是 ImageNet mean/std** |

⚠️ **增强作用在已存成 jpg 的帧序列上**，不是在线解码视频 ⇒ 必须先做 §5.1。

### 5.4 collate_fn 的补帧机制（关键）
```python
left_pad = 6
total_stride = 4
right_pad = ceil(max_len / 4) * 4 - max_len + left_pad
# 补法：首帧 expand(left_pad) + 原序列 + 尾帧 expand(right_pad)
videoLength = ceil(len(vid)/4)*4 + 2*left_pad
```
**⇒ 序列两端各补 6 帧，且总长对齐到 4 的倍数。**
这与 §3 的 `padding=0` 减帧、`/2` 池化配套使用。

---

## 6. 🔴 我之前「复刻」的 4 处错误（对照表）

| # | 我写的 `official_tfnet.py` | 官方真实 | 后果 |
|---|---|---|---|
| 1 | `rfft(x, dim=1)` 沿**时间轴** + 自创零填充回 T | `fft(x, dim=-1)` 沿**特征维** + `abs()`，长度天然不变 | 论文式(3) 被换成另一个东西；我记的「零填充偏差」**根本不存在** |
| 2 | 手写 `Linear(368→256)` 当 F_f | `resnet34MAM()` 3D RGB backbone | 论文核心贡献（抗复杂背景）完全没复现 |
| 3 | 自创 VAE（KL+MSE） | `SeqKD(T=8)`（见 §7） | 完全对不上 |
| 4 | `nn.Linear` 输出层 | `NormLinear`（权重通道归一化） | 未对齐 |
| 5 | `Conv1d(k=3)` 单层 | `Conv1d(k=5,pad=0)+MaxPool` ×2 | 时间维长度完全不同（48→48 vs 48→9） |

---

## 7. ⭐ 损失函数（`Train.py`）

```python
PAD_IDX = 0
ctcLoss = nn.CTCLoss(blank=PAD_IDX, reduction='none', zero_infinity=True)   # ★★★
kld     = DataProcessMoudle.SeqKD(T=8)        # TFNet 分支
mseLoss = nn.MSELoss(reduction="mean")         # 只有 MAM-FSD 分支才有
optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-4)
```

**⇒ TFNet 的两个辅助损失是 `SeqKD`（序列知识蒸馏），不是论文文字里的 VAE。**
`SeqKD` 实现：
```python
def forward(self, prediction_logits, ref_logits, use_blank=True):
    prediction_logits = F.log_softmax(prediction_logits[:,:,1:]/T, -1).view(-1, C-1)
    ref_probs         = F.softmax(ref_logits[:,:,1:]/T, -1).view(-1, C-1)
    return KLDivLoss(reduction='batchmean')(prediction_logits, ref_probs) * T * T
```

### 🔴 `zero_infinity=True` 直接解释了我 P72 的 nan 崩溃
我写的是 `reduction='mean', zero_infinity=False`。
`zero_infinity=False` 时不可对齐样本 loss = inf → 污染 Adam 状态 → **永久 nan**。
（我们实测 L/T max=0.333、0 条不可对齐，所以 inf 不是主要来源，
但**一旦偶发一个越界 batch，mean 会把整个 batch 变成 nan**——
这解释了为什么我逐项排除都找不到原因：**触发罕见，但后果不可逆**。）

---

## 8. 词表（`DataProcessMoudle.py::Word2Id`）

```python
wordList = []
for split in (train, valid, test):        # ⭐ 三个 split 的 gloss 全收
    words = row[3].split("/")
    words = PreWords(words)
    wordList += words
idx2word = [PAD] + sorted(list(set(wordList)))
return word2idx, len(idx2word) - 1, idx2word
```

**⇒ 无 min_freq、无 max_size、无 most_common。只出现 1 次的 gloss 也进词表。**
**⇒ 连 test split 的词都进词表**（词表层面 transductive，但训练只看 train 的帧，
不违反「test 冻结」—— 官方没在 test 上算指标）。

### `PreWords` 的归并规则
```python
if word[j] in "({[（":  subFlag = True                    # 删括号及内容
if word[-1].isdigit() and not word[0].isdigit(): pop()   # 删词尾数字
if word[0] in ",，":  wordList[0] = "，"                   # 英文逗号→中文
if word[0] in "?？":  wordList[0] = "？"                   # 英文问号→中文
if word.isdigit(): word = str(int(word))                # 纯数字去前导零
```
**实测：原始 distinct 3841 → 3515，合并掉 433 个词**
（`一些1`/`一些2`、`一定1`/`一定2`、`上班1`、`不行2`、`专{专业词汇的第一个手势}` 等）。

### 实测词表对比
| 口径 | 词表 |
|---|---|
| 我们（train only, min_freq=1） | 3517 |
| **官方（train+dev+test，无截断）** | **3515** |
| 差异 | 我们多 `<unk>`；**官方有我们无 = 0 个** |

**⇒ 词表口径已对齐，不用改。**

---

## 9. 解码与配置

| 项 | 官方值 | 来源 |
|---|---|---|
| 解码 | **CTC beam search, `beam_width=10`** | `decode.py` `ctcdecode.CTCBeamDecoder` |
| hiddenSize | 1024 | `params/config.ini` |
| lr | 1e-4 | 同上 |
| batchSize | 2 | 同上 |
| epochs | 55（论文） | 论文 Implementation rules |
| lr 衰减 | 35/45 降 80% | 论文 |
| 输出层 | `wordSetNum * max_num_states + 1`，`max_num_states=1` → 3516 | `Train.py:54,103` |
| 硬件 | RTX3090Ti 24GB | 论文 |

`params/config.ini` **没有 `wordSetNum` 配置项**，词表是代码动态算的。

---

## 10. CSV 一致性（✅ 无需改动）

```
official data/CE-CSL/{train,dev,test}.csv   4974 / 516 / 501 行
我们    data/raw/CE-CSL/label/*.csv         4974 / 516 / 501 行
表头一致：Number,Translator,Chinese Sentences,Gloss,Note
```

---

## 11. 官方基准与我们的定位

**CE-CSL Dev/Test WER %（Table VII）**

| 模型 | Dev | Test |
|---|---|---|
| MSTNet | 54.4 | 53.0 |
| CorrNet | 47.2 | 46.5 |
| SEN | 46.5 | 45.3 |
| VAC | 45.1 | 43.3 |
| MAM-FSD | 44.9 | 44.7 |
| **TFNet** | **42.1** | **41.9** |

**我们的历史数字定位**
| 配置 | 特征 | 词表 | WER_official |
|---|---|---|---|
| P42 lm_only | 旧 holistic | **300** | 52.11% |
| P72 TFNet（landmark 版） | Tasks API | 3517 | 76.00% |

⚠️ **P42 与 P72 的分母相同（都是 2842 参考 token、dev unk 0%），
所以两个数直接可比 —— P72 真实退步 23.89 pp，不是"不可比"。**
（我曾用"词表不同所以不可比"解释，实测证伪，已撤回。）

**官方权重在百度网盘（码 0000），本环境取不到**
⇒ 只能从 ImageNet 初始化自己训，**42.1% 大概率达不到**，
但能拿到「官方路线的真实天花板」。

---

## 12. 待办

1. 读 `Module.py` 的 `MotorAttention`（未细读）
2. 实现官方 `TemporalConv`（k=5 + MaxPool）而非单层 Conv1d
3. 实现 `collate_fn` 的 `left_pad=6` 补帧
4. 用 `NormLinear` 替换 `nn.Linear`
5. `zero_infinity=True` + nan 防护（保住 Adam 状态）
6. **`RandomHorizontalFlip(0.5)` 必须做开关对比** —— 中文手语镜像语义可能相反
7. 小规模（dev 515）先跑通 → 再上全量