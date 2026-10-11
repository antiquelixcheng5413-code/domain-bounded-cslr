"""P122: 补齐 best.pt 缺失的 RNG 状态

发现：p119-hflip-best.pt 缺 torch_rng/np_rng/py_rng/cuda_rng，
而 last.pt 有。原因在 p78_train_official_tfnet.py 第 809 行的存盘字典。
⇒ 从 best 续训时 data shuffle / dropout 的随机流会重新初始化。
影响：不改变模型权重，但使「从 best 续训」与「继续训练」不完全等价。
"""
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
p = REPO / "tools/blank_gov/p78_train_official_tfnet.py"
s = p.read_text(encoding="utf-8")

old = '''            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "sched_state": sched.state_dict(),
                        "wordSetNum": wordSetNum, "idx2word": idx2word,
                        "hidden": a.hidden, "hflip": not a.no_hflip,
                        "wer_official": best, "history": hist,
                        "total_epochs_planned": T,
                        "config": vars(a)},
                       ck_dir / ("%s-best.pt" % a.tag))'''

new = '''            # 🔴 best.pt 原本缺 RNG 状态（last.pt 有）⇒ 从 best 续训时
            #    data shuffle / dropout 的随机流会重新初始化。
            #    实例：p119-hflip-best.pt（ep33, 56.72%）就是缺 RNG 存的。
            #    P122 已补齐四项 RNG。
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "sched_state": sched.state_dict(),
                        "torch_rng": torch.random.get_rng_state(),
                        "np_rng": np.random.get_state(),
                        "py_rng": random.getstate(),
                        "cuda_rng": (torch.cuda.get_rng_state_all()
                                     if torch.cuda.is_available() else None),
                        "wordSetNum": wordSetNum, "idx2word": idx2word,
                        "hidden": a.hidden, "hflip": not a.no_hflip,
                        "wer_official": best, "history": hist,
                        "total_epochs_planned": T,
                        "config": vars(a)},
                       ck_dir / ("%s-best.pt" % a.tag))'''

if old not in s:
    print("NOT_FOUND: 目标串不存在，可能已修改过")
    raise SystemExit(1)

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("OK: best.pt 已补齐 torch_rng/np_rng/py_rng/cuda_rng")
