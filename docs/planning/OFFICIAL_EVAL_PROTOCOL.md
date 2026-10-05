{
  "generated": "2026-10-06",
  "trigger": "用户要求『之后模型效果的判断按照官方的来，计算方法也是』",
  "official_source": "arXiv:2409.11960v2 (CE-CSL 官方论文) Section IV-A + 式(11)",

  "official_wer_definition": {
    "formula": "WER = 100% × (ins + del + subs) / sum",
    "paper_text": "where ins represents the number of words to be inserted, del represents the number of words to be deleted, subs represents the number of words to be replaced, and sum represents the total number of words in the label.",
    "denominator": "sum = **整个语料的参考 token 总数**（corpus-level）",
    "level": "语料级（corpus-level），**不是逐句平均**",
    "note": "官方引用 Koller et al. [8] 作为 WER 定义来源 —— CSLR 领域通用做法"
  },

  "punctuation_verified": {
    "method": "用官方 Table VII 的定性样例反推分母",
    "case1": {
      "ref": "他/小孩时间/开始/做/律师/希望/。",
      "hyp": "他/时间/开始/。",
      "official_wer": 50.0,
      "my_naive_split": "7 token -> 4/7 = 57.1%",
      "reverse_inferred": "4/0.50 = **8 token**",
      "conclusion": "**标点「。」计入分母**，且官方把「小孩时间」视作两个 gloss（小孩+时间）"
    },
    "case2_tfnet": {"ref": "他/小孩时间/开始/做/律师/希望/。",
                    "hyp": "同参考", "official_wer": 0.0, "computed": "0.0% ✓"},
    "conclusion": "**标点保留在参考里**（keep_punctuation=True，与GlossSequenceConfig 默认一致）"
  },

  "official_protocol_for_our_project": {
    "分母": "语料级参考 token 总数（我们 dev = 2838）",
    "分子": "语料上全部 ins+del+subs 之和（我们 = 1479）",
    "预处理": "不做任何额外处理 —— 官方论文未定义标点移除/去重/大小写归一/空格处理",
    "参考来源": "用 voc.encode(raw) → voc.decode(ids)（库路径，含 clean_token 归一化）",
    "判优指标": "**只看官方 token 级 WER**",
    "补充诊断": "逐句等权、exact 率、剔 unk WER —— 官方未定义，仅作内部诊断，**不用于对外报数**"
  },

  "verified_our_dev": {
    "official_formula_result": 0.5211,
    "receipt_value": 0.5211,
    "match": true,
    "edit_total": 1479,
    "ref_token_total": 2838,
    "samples": 514,
    "note": "我们一直报的 token 级 0.5211 **本来就是官方口径**，这一条无需修改"
  },

  "🔴 P50 的 2.8774 是算错的 —— 撤回": {
    "错误": "P50 报『逐句等权 WER = 2.8774』，并写成『与 token 级差 5.5 倍』",
    "实测": "mean(edit_i / len(ref_i)) = **0.5128**",
    "根因": "我当时把**编辑总数 1479 除以了句数 514** → 2.8774。那是除错了分母，"
            "不是逐句等权 WER",
    "P50 收据里其实已写下正确描述": "'分母 = ref token 总数 2838（正确）。若误用句数 514 会得到荒谬的 2.8774' —— 我写对了诊断，却把那个荒谬值当成另一个口径报了出来",
    "正确的四口径（本项目实测）": {
      "官方 token级（语料级）": 0.5211,
      "逐句等权mean(edit_i/len(ref_i))": 0.5128,
      "逐句等权中位数": 0.500,
      "exact 率": "18/514 = 3.5%",
      "剔 unk WER": 0.5757,
      "per-sentence WER 范围": "min 0.000 / median 0.500 / max 1.200；>1.0 的只有 1 句(0.2%)"
    },
    "对结论的影响": {
      "被推翻": "『token 级与逐句等权差 5.5 倍，短视频错得最狠』—— **不成立**",
      "仍成立": "『只看 token 级 0.5211 会掩盖样本级真相』—— 靠的是 **exact 率 3.5%**，不是逐句差5.5 倍",
      "仍然有效的现象": "输出短的视频里 unk 占比高（P44 实测 >=50% unk 的句子 129/514），"
                        "因为 unk 挤占短句的 token 空间"
    }
  },

  "new_rules": {
    "R1": "对外报数**只用官方 token 级（语料级）WER**，公式见官方式(11)",
    "R2": "不做任何官方未定义的预处理（不移标点、不去重、不归一化大小写）",
    "R3": "参考序列必须用库路径 voc.encode(raw)→decode，不能自己 split",
    "R4": "与官方表格对照时，只对照 CE-CNSL 列（官方是按数据集分别训练评估，无混合）",
    "R5": "逐句等权 / exact / 剔unk 降级为**内部诊断**，写进收据但不作为判优或对外指标",
    "R6": "test split 仍冻结 —— 只在最终验收用一次；平时判优用 dev"
  },

  "official_benchmarks_ce_csl": {
    "note": "arXiv:2409.11960v2 Table VI，Dev/Test WER (%)",
    "VAC": [45.1, 43.3],
    "MSTNet": [54.4, 53.0],
    "SEN": [46.5, 45.3],
    "CorrNet": [47.2, 46.5],
    "MAM-FSD": [44.9, 44.7],
    "TFNet": [42.1, 41.9]
  },
  "official_training_protocol": {
    "frame_feature": "MAM-FSD 的 CNN backbone",
    "optimizer": "Adam, lr 1e-4, weight decay 1e-4",
    "batch": 2,
    "epochs": "55，lr 在第 35/45 轮降 80%",
    "augmentation": "随机裁剪 256→224 + 水平翻转 0.5 + **时序增强 ±20%**",
    "test_aug": "仅中心裁剪",
    "decoding": "CTC beam search, width 10",
    "hardware": "RTX3090Ti 24GB"
  },

  "implications_for_next_experiment": {
    "vocab": "3516（与官方 3515 吻合），dev OOV = 0% —— 与官方前提一致",
    "beam": "官方用 beam 10，我们当前 greedy —— **必须补上才算可比**",
    "augmentation": "官方三项（裁剪/翻转/时序±20%）我们一项都没测过",
    "judgment": "新结果先报官方 token 级 WER；只要在 dev 上明显低于 0.5211 就是真实改善"
  }
}