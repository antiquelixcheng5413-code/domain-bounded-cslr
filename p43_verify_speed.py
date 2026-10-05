import sys, glob, time
sys.path.insert(0, 'src')
sys.path.insert(0, 'tools/blank_gov')
sys.path.insert(0, 'app/backend')
from pathlib import Path
import ctc_landmark_service as M

svc = M.CtcLandmarkService(Path('artifacts/checkpoints/p42-lm_only-ep100.pt'))
print('ready=', svc.ready, svc.model_error, flush=True)

vids = sorted(glob.glob('/mnt/c/Users/su127/Desktop/csl视频/dev/A/*.mp4'))[:6]
tot = 0.0
for i, v in enumerate(vids):
    pr = svc.predict_video(Path(v))
    t = pr.latency_ms.get('total', 0)
    tot += t
    print('%d %-14s %-5s total=%6.0fms gloss=%r conf=%.4f'
          % (i, v.split('/')[-1], pr.status, t, pr.gloss, pr.confidence), flush=True)
print('mean total = %.2fs' % (tot / len(vids)), flush=True)

print('--- repeatability ---', flush=True)
p = Path(vids[3])
for k in range(3):
    pr = svc.predict_video(p)
    print(' ', k, pr.status, repr(pr.gloss), 'conf=%.4f' % pr.confidence,
          '%.0fms' % pr.latency_ms.get('total', 0), flush=True)
print('DONE', flush=True)
