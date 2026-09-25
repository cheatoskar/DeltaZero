"""Torch half of features v2: block tokens (built on the GPU in training, on the CPU live)
and the DriverNet2 model. The math mirrors ghost.block_features_np; tests/test_ghost.py
checks that both give the same tokens."""
import math

import torch
import torch.nn as nn

from . import ghost as G


def block_features(center, heading, name, wp, tb, valid, pos, psi):
    """center (B,M,3) world m, heading (B,M), name/wp (B,M) long, tb (B,M) tie-break,
    valid (B,M) bool, pos (B,3), psi (B,) -> names (B,K), wps (B,K), feats (B,K,F), mask (B,K)."""
    B, M = name.shape
    K = G.K_BLOCKS
    dev = center.device
    rel = center - pos[:, None, :]
    dist = torch.linalg.vector_norm(rel, dim=-1)
    key = (torch.floor(dist * 64.0) + tb).masked_fill(~valid, float('inf'))
    k = min(K, M)
    names = torch.zeros(B, K, dtype=torch.long, device=dev)
    wps = torch.zeros(B, K, dtype=torch.long, device=dev)
    feats = torch.zeros(B, K, G.BLOCK_FEAT, dtype=center.dtype, device=dev)
    mask = torch.zeros(B, K, dtype=torch.bool, device=dev)
    if k == 0:
        return names, wps, feats, mask
    _, order = torch.topk(key, k, dim=1, largest=False, sorted=True)
    d = torch.gather(dist, 1, order).masked_fill(~torch.gather(valid, 1, order), float('inf'))
    ok = d <= G.BLOCK_RANGE
    r = torch.gather(rel, 1, order[..., None].expand(-1, -1, 3))
    s, c = torch.sin(psi)[:, None], torch.cos(psi)[:, None]
    x = r[..., 0] * c - r[..., 2] * s            # right = (cos, 0, -sin)
    z = r[..., 0] * s + r[..., 2] * c            # fwd   = (sin, 0, cos)
    rh = torch.gather(heading, 1, order) - psi[:, None]
    f = torch.stack([x / 64.0, r[..., 1] / 64.0, z / 64.0, torch.sin(rh), torch.cos(rh),
                     torch.where(ok, d, torch.zeros_like(d)) / G.BLOCK_RANGE], -1)
    f = f * ok[..., None]
    names[:, :k] = torch.where(ok, torch.gather(name, 1, order), 0)
    wps[:, :k] = torch.where(ok, torch.gather(wp, 1, order), 0)
    feats[:, :k] = f
    mask[:, :k] = ok
    return names, wps, feats, mask


class BlockTable:
    """All maps' blocks as flat tensors (CSR) on one device, for batched gathering."""

    def __init__(self, off, xyz, dirs, names, wps, device, max_blocks: int = 1024):
        self.off = torch.as_tensor(off[:-1], dtype=torch.long, device=device)
        self.cnt = torch.as_tensor(off[1:] - off[:-1], dtype=torch.long, device=device)
        self.xyz = torch.as_tensor(xyz, dtype=torch.int16, device=device)
        self.heading = torch.as_tensor(G.dir_heading(dirs), dtype=torch.float32, device=device)
        # compact on the device (~200M blocks for all parts); widened to long after gathering
        self.name = torch.as_tensor(names, dtype=torch.int32, device=device)
        self.wp = torch.as_tensor(wps, dtype=torch.uint8, device=device)
        self.tb = torch.as_tensor(G.block_tiebreak(names, xyz), dtype=torch.float32, device=device)
        self.size = torch.tensor(G.BLOCK_SIZE, dtype=torch.float32, device=device)
        self.max_blocks = max_blocks

    def gather(self, maps):
        cnt = self.cnt[maps]
        M = int(min(max(int(cnt.max()), 1), self.max_blocks))
        ar = torch.arange(M, device=maps.device)
        valid = ar[None, :] < cnt[:, None]
        idx = torch.where(valid, self.off[maps][:, None] + ar[None, :], torch.zeros_like(ar)[None, :])
        center = (self.xyz[idx].float() + 0.5) * self.size
        return center, self.heading[idx], self.name[idx].long(), self.wp[idx].long(), self.tb[idx], valid


class DriverNet2(nn.Module):
    """Transformer over [state | 18 route points | 32 nearby blocks]; heads read the state
    token. steer: STEER_BINS logits (a distribution: keyboard driving is multimodal);
    gas, brake: logits; value: seconds to the finish / 10 (later the RL critic)."""

    def __init__(self, n_names: int, d: int = 256, layers: int = 6, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.cfg = dict(n_names=n_names, d=d, layers=layers, heads=heads, dropout=dropout)
        self.state_in = nn.Sequential(nn.Linear(G.STATE_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.route_in = nn.Linear(3, d)
        self.route_pos = nn.Parameter(torch.zeros(len(G.ROUTE_D), d))
        self.name_emb = nn.Embedding(n_names, d, padding_idx=0)
        self.wp_emb = nn.Embedding(8, d, padding_idx=0)
        self.block_in = nn.Linear(G.BLOCK_FEAT, d)
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.steer = nn.Linear(d, G.STEER_BINS)
        self.pedals = nn.Linear(d, 2)
        self.value = nn.Linear(d, 1)
        nn.init.normal_(self.route_pos, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02)

    def forward(self, state, route, bname, bwp, bfeat, bmask):
        B = state.shape[0]
        s = self.state_in(state).unsqueeze(1) + self.type_emb[0]
        r = self.route_in(route) + self.route_pos + self.type_emb[1]
        b = self.name_emb(bname) + self.wp_emb(bwp) + self.block_in(bfeat) + self.type_emb[2]
        x = torch.cat([s, r, b], 1)
        pad = torch.cat([torch.zeros(B, 1 + r.shape[1], dtype=torch.bool, device=state.device), ~bmask], 1)
        h = self.norm(self.encoder(x, src_key_padding_mask=pad)[:, 0])
        p = self.pedals(h)
        return {'steer': self.steer(h), 'gas': p[:, 0], 'brake': p[:, 1], 'value': self.value(h).squeeze(-1),
                'h': h}       # the state token after the encoder (RL: the critic reads it)


def n_params(m) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6
