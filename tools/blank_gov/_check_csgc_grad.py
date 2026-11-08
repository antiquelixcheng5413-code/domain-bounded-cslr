"""验证 CSGC 损失非零且可反传（避免静默空操作）。"""
import sys
from pathlib import Path
import torch

R = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (R / "tools/blank_gov", R / "external/TFNet", R / "src"):
    sys.path.insert(0, str(p))
from csgc_module import GlossPrototypeBank  # noqa

torch.manual_seed(0)
C, D, T, B = 3516, 1024, 20, 2
cw = torch.randn(D, C)
bank = GlossPrototypeBank(cw, momentum=0.99, tau=0.07, n_neg=16)
main_lp = torch.log_softmax(torch.randn(T, B, C), dim=-1).requires_grad_(True)
feat = torch.randn(T, B, D, requires_grad=True)
tgt = torch.tensor([5, 9, 12, 100, 7, 11])
tgt_len = torch.tensor([4, 2])
lgt = torch.tensor([20, 18])

loss = bank(main_lp, feat, tgt, tgt_len, lgt)
loss.backward()
print("loss =", float(loss.detach()))
print("grad feat norm =", float(feat.grad.norm()))
print("grad main_lp norm =", float(main_lp.grad.norm()))
p0 = bank.proto[5].clone()
print("proto[5] norm =", float(p0.norm()), "finite =", bool(torch.isfinite(bank.proto).all()))
assert float(loss.detach()) > 0, "loss is zero -> silent no-op"
assert float(feat.grad.norm()) > 0, "no gradient to features"
assert float(main_lp.grad.norm()) > 0, "no gradient to CTC logits"
print("CSGC gradient self-check PASS")
