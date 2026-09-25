"""Behaviour cloning on data/m1/dataset.npz (CPU-friendly; GPU if available).

Validation is on held-out MAPS (never seen in training), so the numbers measure
generalisation, not memorisation. The checkpoint carries everything the live driver
needs to rebuild the exact same inputs: vocab, feature version, conventions.
"""
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as TF

from . import features as F
from .collect import M1
from .dataset import DATASET
from .model import DriverNet
from .paths import RUNS

CKPT = RUNS / 'm1' / 'driver.pt'
CKPT_LAST = RUNS / 'm1' / 'driver_last.pt'
KEYS = ('state', 'route', 'bname', 'bwp', 'bfeat', 'bmask')


def load(device):
    z = np.load(DATASET)
    meta = json.loads(str(z['meta']))
    t = {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in z.files if k != 'meta'}
    for k in ('route', 'bfeat', 'state'):
        t[k] = t[k].float()
    for k in ('bname', 'bwp', 'y_steer'):
        t[k] = t[k].long()
    t['y_gas'], t['y_brake'] = t['y_gas'].float(), t['y_brake'].float()
    return t, meta


def batch(t, idx):
    return [t[k][idx] for k in KEYS]


def losses(out, t, idx):
    ce = TF.cross_entropy(out['steer'], t['y_steer'][idx], label_smoothing=0.05)
    gas = TF.binary_cross_entropy_with_logits(out['gas'], t['y_gas'][idx])
    brake = TF.binary_cross_entropy_with_logits(out['brake'], t['y_brake'][idx])
    value = TF.smooth_l1_loss(out['value'], t['y_value'][idx] / 10.0)
    return ce + 0.5 * gas + 0.5 * brake + 0.1 * value, (ce, gas, brake, value)


@torch.no_grad()
def evaluate(model, t, idx, bs=2048):
    model.eval()
    agg = {'loss': 0.0, 'steer_acc': 0.0, 'steer_dir_acc': 0.0, 'steer_mae': 0.0, 'gas_acc': 0.0,
           'brake_recall': 0.0, 'brake_prec': 0.0, 'value_mae_s': 0.0}
    n = 0
    tp = fp = fn = 0
    mid = F.STEER_BINS // 2
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        out = model(*batch(t, j))
        loss, _ = losses(out, t, j)
        pb = out['steer'].argmax(-1)
        yb = t['y_steer'][j]
        agg['loss'] += loss.item() * len(j)
        agg['steer_acc'] += (pb == yb).float().sum().item()
        agg['steer_dir_acc'] += (torch.sign(pb - mid) == torch.sign(yb - mid)).float().sum().item()
        agg['steer_mae'] += (pb - yb).abs().float().sum().item() * 2.0 / (F.STEER_BINS - 1)
        agg['gas_acc'] += ((out['gas'] > 0).float() == t['y_gas'][j]).float().sum().item()
        pbk, ybk = out['brake'] > 0, t['y_brake'][j] > 0.5
        tp += (pbk & ybk).sum().item()
        fp += (pbk & ~ybk).sum().item()
        fn += (~pbk & ybk).sum().item()
        agg['value_mae_s'] += (out['value'] * 10.0 - t['y_value'][j]).abs().sum().item()
        n += len(j)
    for k in list(agg):
        if k not in ('brake_recall', 'brake_prec'):
            agg[k] /= max(n, 1)
    agg['brake_recall'] = tp / max(tp + fn, 1)
    agg['brake_prec'] = tp / max(tp + fp, 1)
    model.train()
    return agg


def train(minutes: float = 12.0, bs: int = 512, lr: float = 2e-3, d: int = 96, layers: int = 2,
          seed: int = 0, log=print):
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t, meta = load(device)
    hold = t['holdout'].numpy()
    tr = np.nonzero(~hold)[0]
    va = np.nonzero(hold)[0]
    if len(va) == 0:
        raise RuntimeError('no held-out maps in the dataset')
    log(f"train {len(tr)} samples / val {len(va)} samples on held-out maps; device {device}")
    model = DriverNet(len(meta['vocab']) + 2, d=d, layers=layers)
    log(f'params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # time the first steps to fix the schedule length from the budget
    rng = np.random.default_rng(seed)
    t0 = time.time()
    probe = 20
    for _ in range(probe):
        j = torch.from_numpy(rng.choice(tr, bs))
        loss, _ = losses(model(*batch(t, j)), t, j)
        opt.zero_grad(); loss.backward(); opt.step()
    per_step = (time.time() - t0) / probe
    total = max(200, int(minutes * 60 / per_step * 0.85))
    log(f'{per_step * 1000:.0f} ms/step -> {total} steps (~{total * bs / len(tr):.1f} epochs) in {minutes} min')
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    best, history = None, []
    eval_every = max(100, total // 8)
    perm = np.array([], dtype=np.int64)
    for step in range(total):
        if len(perm) < bs:
            perm = rng.permutation(tr)
        j, perm = torch.from_numpy(perm[:bs]), perm[bs:]
        loss, parts = losses(model(*batch(t, j)), t, j)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % eval_every == 0 or step + 1 == total:
            m = evaluate(model, t, torch.from_numpy(va))
            m['step'] = step + 1
            m['train_loss'] = loss.item()
            history.append(m)
            log(f"step {step + 1:5d}  train {loss.item():.3f}  val {m['loss']:.3f}  steer acc {m['steer_acc']:.3f} "
                f"dir {m['steer_dir_acc']:.3f}  gas {m['gas_acc']:.3f}  brake P/R {m['brake_prec']:.2f}/{m['brake_recall']:.2f}  "
                f"value MAE {m['value_mae_s']:.2f}s")
            if best is None or m['loss'] < best['loss']:
                best = m
                CKPT.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'state_dict': model.state_dict(), 'cfg': model.cfg, 'meta': meta,
                            'features_version': F.FEATURES_VERSION, 'val': m}, CKPT)
    # The final weights too: best-on-held-out-maps (driver.pt) measures generalisation, but
    # on maps that WERE in training the later, better-fitted weights drive better.
    torch.save({'state_dict': model.state_dict(), 'cfg': model.cfg, 'meta': meta,
                'features_version': F.FEATURES_VERSION, 'val': history[-1]}, CKPT_LAST)
    (RUNS / 'm1' / 'train_history.json').write_text(json.dumps(history, indent=1), encoding='utf-8')
    log(f"best val loss {best['loss']:.3f} at step {best['step']} -> {CKPT}")
    return best
