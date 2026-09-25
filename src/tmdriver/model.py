"""DriverNet: a small transformer over [state | route points | nearby blocks].

Outputs (read from the state token):
  steer  STEER_BINS logits over [-1, 1]; the end bins are full lock, i.e. a key press.
         A distribution, not a regression: keyboard driving is multimodal (left / none /
         right), and the mean of that would be a steer nobody ever pressed.
  gas, brake   logits
  value  seconds until the finish (auxiliary; later the RL critic)
"""
import torch
import torch.nn as nn

from . import features as F


class DriverNet(nn.Module):
    def __init__(self, n_names: int, d: int = 96, layers: int = 2, heads: int = 4):
        super().__init__()
        self.cfg = dict(n_names=n_names, d=d, layers=layers, heads=heads)
        self.state_in = nn.Sequential(nn.Linear(F.STATE_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.route_in = nn.Linear(3, d)
        self.route_pos = nn.Parameter(torch.zeros(len(F.ROUTE_D), d))
        self.name_emb = nn.Embedding(n_names, d, padding_idx=0)
        self.wp_emb = nn.Embedding(6, d, padding_idx=0)
        self.block_in = nn.Linear(F.BLOCK_FEAT, d)
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.steer = nn.Linear(d, F.STEER_BINS)
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
        pedals = self.pedals(h)
        return {'steer': self.steer(h), 'gas': pedals[:, 0], 'brake': pedals[:, 1],
                'value': self.value(h).squeeze(-1)}
