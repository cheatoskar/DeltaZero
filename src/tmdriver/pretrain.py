"""Stage A: behaviour cloning of the ghost-feature driver on the trace shards.

    python tmdriver.py pretrain --hours 5 [--d 256 --layers 6] [--init runs/ghost/latest.pt]

Data: every part in data/ghost/shards with a DONE marker. A batch is CHUNKS contiguous
slices of the (pre-shuffled) train arrays of random parts, so reads stay sequential while
the batch mixes ~16 x 2000 maps. Block tokens are built on the device from the block
table. Validation is on held-out MAPS only (track-id hash), so it measures driving on
maps the model has never seen.

Checkpoints (runs/ghost/): latest.pt every --save-min minutes, best.pt on validation
loss, both loadable by the live driver (kind = 'ghost2').
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

from . import ghost as G
from .ghost_torch import BlockTable, DriverNet2, block_features, n_params
from .paths import RUNS
from .tracebuild import SHARDS, VOCAB

OUT = RUNS / 'ghost'
FIELDS = ('state', 'route', 'pos', 'psi', 'map', 'steer', 'pedal', 'value', 'rline')


def parse_sources(spec: str):
    """'dirA:0.7,dirB:0.3' -> [(Path, weight)] (weight optional, default 1); empty -> the
    trace shards alone. Only a numeric suffix counts as a weight (Windows drive letters)."""
    if not spec:
        return [(SHARDS, 1.0)]
    out = []
    for item in spec.split(','):
        path, sep, w = item.rpartition(':')
        try:
            out.append((Path(path), float(w)) if sep else (Path(item), 1.0))
        except ValueError:
            out.append((Path(item), 1.0))
    return out


class Shards:
    """Memory-mapped train/val arrays of all finished parts of one or more shard roots,
    plus one global block table. Each root gets a share of the batches (its weight),
    spread over its parts by size, so a small replay set is not drowned by the traces."""

    def __init__(self, roots=None, split: str = 'train', max_parts: int = 0):
        roots = roots or [(SHARDS, 1.0)]
        self.parts, self.sizes, self.map_base, self.weights, self.root_of = [], [], [], [], []
        offs, xyz, dirs, names, wps = [np.zeros(1, np.int64)], [], [], [], []
        base = 0
        nb = 0
        plan = []
        for ri, (root, w) in enumerate(roots):
            parts = sorted(p for p in Path(root).glob('part*') if (p / 'DONE').exists())
            if max_parts:
                parts = parts[:max_parts]
            if not parts:
                raise RuntimeError(f'no finished shards in {root}')
            plan += [(ri, w, p) for p in parts]
        for ri, w, p in plan:
            off = np.load(p / 'b_off.npy')
            offs.append(off[1:] + nb)
            nb += int(off[-1])
            xyz.append(np.load(p / 'b_xyz.npy'))
            dirs.append(np.load(p / 'b_dir.npy'))
            names.append(np.load(p / 'b_name.npy'))
            wps.append(np.load(p / 'b_wp.npy'))
            f = p / f'{split}_map.npy'
            n = len(np.load(f, mmap_mode='r')) if f.exists() else 0
            self.parts.append(p)
            self.sizes.append(n)
            self.map_base.append(base)
            self.root_of.append(ri)
            self.weights.append(w)
            base += len(off) - 1
        sizes = np.array(self.sizes, np.float64)
        prob = np.zeros(len(sizes))
        for ri in set(self.root_of):
            m = np.array(self.root_of) == ri
            if sizes[m].sum() > 0:
                prob[m] = self.weights[int(np.nonzero(m)[0][0])] * sizes[m] / sizes[m].sum()
        self.prob = prob / prob.sum() if prob.sum() > 0 else prob
        self.block_off = np.concatenate(offs)
        self.block_xyz = np.concatenate(xyz)
        self.block_dir = np.concatenate(dirs)
        self.block_name = np.concatenate(names)
        self.block_wp = np.concatenate(wps)
        self.split = split
        self.n = int(sum(self.sizes))
        self.n_maps = base

    def arrays(self, i):
        p = self.parts[i]
        return {k: np.load(p / f'{self.split}_{k}.npy', mmap_mode='r') for k in FIELDS}

    def block_table(self, device, max_blocks=1024):
        return BlockTable(self.block_off, self.block_xyz, self.block_dir, self.block_name, self.block_wp,
                          device, max_blocks)


class BatchStream(torch.utils.data.IterableDataset):
    def __init__(self, shards: Shards, bs: int, chunk: int, seed: int):
        self.sh, self.bs, self.chunk, self.seed = shards, bs, chunk, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 7919)
        prob = self.sh.prob
        cache = {}
        while True:
            parts = rng.choice(len(prob), self.bs // self.chunk, p=prob)
            out = {k: [] for k in FIELDS}
            gmap = []
            for i in parts:
                a = cache.pop(i, None)
                if a is None:
                    a = self.sh.arrays(i)
                cache[i] = a                       # most recent last
                if len(cache) > 48:                # 9 memmaps per part: stay far below the fd limit
                    cache.pop(next(iter(cache)))
                n = self.sh.sizes[i]
                s = int(rng.integers(0, max(1, n - self.chunk)))
                for k in FIELDS:
                    out[k].append(np.asarray(a[k][s:s + self.chunk]))
                gmap.append(np.asarray(a['map'][s:s + self.chunk]).astype(np.int64) + self.sh.map_base[i])
            b = {k: torch.from_numpy(np.concatenate(v)) for k, v in out.items()}
            b['gmap'] = torch.from_numpy(np.concatenate(gmap))
            yield b


def load_val(shards: Shards, max_n: int, seed: int = 1):
    """Held-out samples from the FIRST shard root (the data the run is about)."""
    rng = np.random.default_rng(seed)
    first = [i for i in range(len(shards.parts)) if shards.root_of[i] == 0 and shards.sizes[i] > 0]
    per = max(1, max_n // max(1, len(first)))
    out = {k: [] for k in FIELDS}
    gmap = []
    for i in first:
        a = shards.arrays(i)
        n = shards.sizes[i]
        s = int(rng.integers(0, max(1, n - per)))
        for k in FIELDS:        # np.array copies: a view would keep every part's memmap open
            out[k].append(np.array(a[k][s:s + per]))
        gmap.append(np.array(a['map'][s:s + per]).astype(np.int64) + shards.map_base[i])
        del a
    b = {k: torch.from_numpy(np.concatenate(v)) for k, v in out.items()}
    b['gmap'] = torch.from_numpy(np.concatenate(gmap))
    return b


def prepare(b, table: BlockTable, device, route_drop: float = 0.0, gen=None, orient: bool = False,
            orient_drop: float = 0.0):
    """Batch dict (CPU) -> model inputs + targets on device. Without `orient` the orientation
    group is always blanked (a model that never learnt it must not see it live either)."""
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    state = b['state'].float()
    route = b['route'].float()
    if route_drop > 0:
        drop = torch.rand(state.shape[0], device=device, generator=gen) < route_drop
        route = route * (~drop)[:, None, None]
        state[:, G.S_ROUTE_FLAG] = state[:, G.S_ROUTE_FLAG] * (~drop)
    if not orient:
        state[:, G.S_ORIENT] = 0.0
        state[:, G.S_ORIENT_FLAG] = 0.0
    elif orient_drop > 0:
        keep = (torch.rand(state.shape[0], device=device, generator=gen) >= orient_drop).float()
        state[:, G.S_ORIENT] = state[:, G.S_ORIENT] * keep[:, None]
        state[:, G.S_ORIENT_FLAG] = state[:, G.S_ORIENT_FLAG] * keep
    center, heading, name, wp, tb, valid = table.gather(b['gmap'])
    bn, bw, bf, bm = block_features(center, heading, name, wp, tb, valid, b['pos'].float(), b['psi'].float())
    x = (state, route, bn, bw, bf, bm)
    y = {'steer': b['steer'].long(), 'gas': (b['pedal'] & 1).float(), 'brake': ((b['pedal'] >> 1) & 1).float(),
         'value': b['value'].float(), 'rline': b['rline']}
    return x, y


def losses(out, y):
    ce = TF.cross_entropy(out['steer'].float(), y['steer'], label_smoothing=0.05)
    gas = TF.binary_cross_entropy_with_logits(out['gas'].float(), y['gas'])
    brake = TF.binary_cross_entropy_with_logits(out['brake'].float(), y['brake'])
    value = TF.smooth_l1_loss(out['value'].float(), y['value'] / 10.0)
    return ce + 0.5 * gas + 0.5 * brake + 0.1 * value


@torch.no_grad()
def evaluate(model, val, table, device, amp, bs=8192, orient=False):
    model.eval()
    n = len(val['map'])
    acc = {'loss': 0.0, 'steer_acc': 0.0, 'steer_dir_acc': 0.0, 'gas_acc': 0.0, 'value_mae_s': 0.0}
    tp = fp = fn = 0
    line_ok = line_n = noline_ok = noline_n = 0
    mid = G.STEER_BINS // 2
    for i in range(0, n, bs):
        b = {k: v[i:i + bs] for k, v in val.items()}
        x, y = prepare(b, table, device, orient=orient)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(*x)
        m = len(y['steer'])
        acc['loss'] += losses(out, y).item() * m
        pb = out['steer'].float().argmax(-1)
        hit = (pb == y['steer'])
        acc['steer_acc'] += hit.float().sum().item()
        acc['steer_dir_acc'] += (torch.sign(pb - mid) == torch.sign(y['steer'] - mid)).float().sum().item()
        acc['gas_acc'] += ((out['gas'] > 0).float() == y['gas']).float().sum().item()
        acc['value_mae_s'] += (out['value'].float() * 10 - y['value']).abs().sum().item()
        pk, yk = out['brake'] > 0, y['brake'] > 0.5
        tp += (pk & yk).sum().item(); fp += (pk & ~yk).sum().item(); fn += (~pk & yk).sum().item()
        rl = y['rline'] > 0
        line_ok += hit[rl].sum().item(); line_n += rl.sum().item()
        noline_ok += hit[~rl].sum().item(); noline_n += (~rl).sum().item()
    for k in acc:
        acc[k] /= max(n, 1)
    acc['brake_prec'] = tp / max(tp + fp, 1)
    acc['brake_recall'] = tp / max(tp + fn, 1)
    acc['steer_acc_line'] = line_ok / max(line_n, 1)
    acc['steer_acc_noline'] = noline_ok / max(noline_n, 1)
    acc['n'] = n
    model.train()
    return acc


def save(path, model, meta, extra):
    tmp = path.with_suffix('.tmp')
    torch.save(dict({'kind': 'ghost2', 'features_version': G.FEATURES_VERSION,
                     'state_dict': model.state_dict(), 'cfg': model.cfg, 'meta': meta}, **extra), tmp)
    tmp.replace(path)


def pretrain(hours: float = 1.0, bs: int = 4096, lr: float = 1e-3, d: int = 256, layers: int = 6, heads: int = 8,
             chunk: int = 256, workers: int = 6, route_drop: float = 0.25, eval_min: float = 10.0,
             save_min: float = 15.0, val_n: int = 200_000, max_parts: int = 0, init: str = '',
             compile_model: bool = False, seed: int = 0, shards: str = '', orient: bool = False,
             orient_drop: float = 0.3, out: str = '', log=print):
    """shards: 'root:weight,...' (default: the trace shards). orient: learn from the car
    orientation where the data has it (replays; the live driver then feeds it too).
    out: checkpoint folder (default runs/ghost)."""
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda'
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    OUT = Path(out) if out else RUNS / 'ghost'
    OUT.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(VOCAB.read_text(encoding='utf-8'))
    roots = parse_sources(shards)
    train = Shards(roots, split='train', max_parts=max_parts)
    val_sh = Shards(roots, split='val', max_parts=max_parts)
    table = train.block_table(device)
    val = load_val(val_sh, val_n)
    log(f"sources: {', '.join(f'{r} x{w}' for r, w in roots)}; orientation {'on' if orient else 'off'}")
    log(f'{len(train.parts)} parts, {train.n:,} train samples, {len(val["map"]):,} val samples '
        f'(held-out maps), {train.n_maps:,} maps, {len(train.block_name):,} blocks; device {device}')

    model = DriverNet2(len(vocab) + 2, d=d, layers=layers, heads=heads).to(device)
    start_step, seen = 0, 0
    if init:
        ck = torch.load(init, map_location=device, weights_only=False)
        if ck['meta']['vocab'] != vocab:
            raise RuntimeError(f'{init} was trained with another block vocabulary')
        model.load_state_dict(ck['state_dict'])
        log(f'initialised from {init}')
    log(f'model {n_params(model):.2f}M params, d={d} layers={layers} heads={heads}')
    fwd = torch.compile(model) if compile_model else model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    meta = {'vocab': vocab, 'route_drop': route_drop, 'orient_used': bool(orient), 'orient_drop': orient_drop,
            'decide_every_ms': 50, 'data': [f'{r}:{w}' for r, w in roots], 'init': init}

    dl = torch.utils.data.DataLoader(BatchStream(train, bs, chunk, seed), batch_size=None,
                                     num_workers=workers, pin_memory=device.type == 'cuda',
                                     prefetch_factor=4 if workers else None, persistent_workers=bool(workers))
    it = iter(dl)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # measure the step time, then fix the cosine schedule to the time budget
    t_start = time.time()
    probe = 30
    for _ in range(probe):
        x, y = prepare(next(it), table, device, route_drop, gen, orient, orient_drop)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss = losses(fwd(*x), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    per = (time.time() - t_start) / probe
    total = max(500, int(hours * 3600 / per * 0.92))
    warm = min(2000, total // 20)
    log(f'{per * 1000:.0f} ms/step ({bs / per:,.0f} samples/s) -> {total:,} steps, '
        f'{total * bs / max(train.n, 1):.2f} epochs in {hours} h')
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))))

    best = None
    hist = []
    t_eval = t_save = time.time()
    t_log = time.time()
    run_loss = 0.0
    for step in range(total):
        x, y = prepare(next(it), table, device, route_drop, gen, orient, orient_drop)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss = losses(fwd(*x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        seen += bs
        run_loss = 0.98 * run_loss + 0.02 * loss.item() if step else loss.item()
        now = time.time()
        if now - t_log > 60:
            log(f'step {step + 1:,}/{total:,}  loss {run_loss:.3f}  lr {sched.get_last_lr()[0]:.2e}  '
                f'{seen / (now - t_start):,.0f} samples/s  {(now - t_start) / 3600:.2f} h')
            t_log = now
        last = step + 1 == total
        if now - t_eval > eval_min * 60 or last:
            m = evaluate(model, val, table, device, amp, orient=orient)
            m.update(step=step + 1, samples=seen, hours=round((now - t_start) / 3600, 3), train_loss=run_loss)
            hist.append(m)
            log(f"VAL step {step + 1:,}: loss {m['loss']:.3f}  steer acc {m['steer_acc']:.3f} "
                f"(line {m['steer_acc_line']:.3f} / none {m['steer_acc_noline']:.3f})  dir {m['steer_dir_acc']:.3f}  "
                f"gas {m['gas_acc']:.3f}  brake P/R {m['brake_prec']:.2f}/{m['brake_recall']:.2f}  "
                f"value MAE {m['value_mae_s']:.2f}s")
            if best is None or m['loss'] < best['loss']:
                best = m
                save(OUT / 'best.pt', model, meta, {'val': m, 'step': step + 1})
            (OUT / 'history.json').write_text(json.dumps(hist, indent=1), encoding='utf-8')
            t_eval = time.time()
        if time.time() - t_save > save_min * 60 or last:
            save(OUT / 'latest.pt', model, meta, {'val': hist[-1] if hist else {}, 'step': step + 1})
            t_save = time.time()
    log(f"done: best val loss {best['loss']:.3f} at step {best['step']:,} -> {OUT / 'best.pt'}")
    return best
