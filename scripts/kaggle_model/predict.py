"""Five-seed ResGRU-UNet forecast for the challenge test set. Writes preds.npy (218, 140, 124, 16)."""
import os, sys, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study as S

SEEDS = [42, 1337, 1, 2, 3]
AB = 4

ap = argparse.ArgumentParser()
ap.add_argument('--no-cal', action='store_true', help='skip the 16 calibration factors')
cal = not ap.parse_args().no_cal

S.C.V4_DIR = HERE
stats = S.build_stats()
_, static = S.load_months(stats)
test_ds = S.PM25Test(stats, static)
preds = []
for s in SEEDS:
    ck = torch.load(os.path.join(HERE, f'resgru_s{s}_phaseA_best.pt'), map_location=S.DEVICE, weights_only=False)
    model = S.ResGRUUNet(S.C.C_IN).to(S.DEVICE)
    model.load_state_dict(ck['model'])
    preds.append(S.predict(model, test_ds, AB))
    print(f'seed {s}: done')
p = np.stack(preds).mean(0)
if cal:
    p = p * np.load(os.path.join(HERE, 'horizon_scales.npy'))[None, None, None, :]
p = p.astype(np.float32)
assert p.shape == (S.C.N_TEST, S.C.H, S.C.W, S.C.T_OUT) and np.isfinite(p).all() and p.min() >= 0
np.save(os.path.join(S.C.WORK, 'preds.npy'), p)
print(f'preds.npy {p.shape}, calibration {"on" if cal else "off"}, range [{p.min():.2f}, {p.max():.2f}], mean {p.mean():.3f}')
