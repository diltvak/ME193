"""The PyTorch model and its training loop.

StepNet is a small dilated 1-D CNN (a non-causal temporal conv net) that maps the IMU stream to,
at every 10 ms sample, two numbers:
    step logit   -> sigmoid is the probability that a stride peak of the sensor leg is here
    stride len   -> metres covered by the stride this sample belongs to

Non-causal on purpose: deciding "a stride peaked *here*" needs to see the shape before and after
(+/-0.62 s receptive field). The live counter pays for that with ~0.65 s of latency.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import StepConfig
from .data import N_FEATURES, gaussian_target, make_features


class Block(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2)
        self.norm = nn.GroupNorm(4, ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(F.gelu(self.norm(self.conv(x))))


class StepNet(nn.Module):
    def __init__(self, cfg: StepConfig):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Conv1d(N_FEATURES, cfg.hidden, 1)
        self.blocks = nn.Sequential(*[Block(cfg.hidden, cfg.kernel, d, cfg.dropout) for d in cfg.dilations])
        self.step_head = nn.Conv1d(cfg.hidden, 1, 1)
        self.len_head = nn.Conv1d(cfg.hidden, 1, 1)

    def forward(self, x):
        """x: (B, N_FEATURES, T) -> (step logits (B, T), stride length in m (B, T))."""
        h = self.blocks(self.inp(x))
        logits = self.step_head(h).squeeze(1)
        length = self.cfg.len_center_m + self.cfg.len_scale_m * self.len_head(h).squeeze(1)
        return logits, length


# ----------------------------------------------------------------------------
# training data: random crops with augmentation
# ----------------------------------------------------------------------------
def _crop(rec, cfg: StepConfig, rng):
    """One augmented training crop -> (features (n,8), target (n,), length (n,), len_mask scalar)."""
    n, fs = cfg.crop_n, cfg.fs
    sl = rec["sl"]
    X, peaks = rec["X"][sl], rec["peaks"] - sl.start
    peaks = peaks[(peaks >= 0) & (peaks < len(X))]
    warp = 1.0
    if rng.random() < cfg.warp_prob:
        warp = rng.uniform(*cfg.warp_range)
    need = int(np.ceil(n * warp))
    if len(X) < need:
        warp, need = 1.0, n
    if len(X) < need:                                          # recording shorter than a crop
        X = np.vstack([X, np.repeat(X[-1:], need - len(X), axis=0)])
    start = int(rng.integers(0, len(X) - need + 1))
    seg = X[start:start + need].copy()
    pk = peaks - start
    if warp != 1.0:                                            # time-stretch: >1 speeds the walk up
        src = np.arange(n) * warp
        seg = np.column_stack([np.interp(src, np.arange(need), seg[:, c]) for c in range(6)])
        pk = pk / warp
    pk = pk[(pk >= 0) & (pk < n)]
    seg[:, :3] *= rng.uniform(*cfg.amp_range)
    seg[:, 3:] *= rng.uniform(*cfg.amp_range)
    seg[:, :3] += rng.normal(0, cfg.noise_acc_g, (n, 3))
    seg[:, 3:] += rng.normal(0, cfg.noise_gyro_dps, (n, 3))
    if rng.random() < cfg.drop_prob:                           # mimic a dropped-packet gap
        a = int(rng.integers(0, n - 30))
        b = a + int(rng.integers(5, 26))
        for c in range(6):
            seg[a:b, c] = np.interp(np.arange(a, b), [a - 1, b], [seg[a - 1, c], seg[min(b, n - 1), c]])
    feats = make_features(seg)
    target = gaussian_target(pk, n, cfg.peak_sigma_s * fs)
    use_len = rec["L"] is not None and warp == 1.0
    length = np.full(n, rec["L"] if use_len else 0.0, np.float32)
    return feats, target, length, float(use_len)


def make_batch(pos, neg, cfg: StepConfig, rng):
    crops = []
    for _ in range(cfg.batch):
        pool = neg if (neg and rng.random() < cfg.negative_share) else pos
        crops.append(_crop(pool[int(rng.integers(len(pool)))], cfg, rng))
    x = torch.from_numpy(np.stack([c[0] for c in crops])).transpose(1, 2).contiguous()
    t = torch.from_numpy(np.stack([c[1] for c in crops]))
    L = torch.from_numpy(np.stack([c[2] for c in crops]))
    m = torch.tensor([c[3] for c in crops])
    return x, t, L, m


def train_net(pos, neg, cfg: StepConfig, seed=0, verbose=False):
    """Train a StepNet on `pos` (walking records with peaks) and `neg` (non-walking records)."""
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    rng = np.random.default_rng(seed)
    net = StepNet(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.iters)
    net.train()
    for it in range(cfg.iters):
        x, target, length, mask = make_batch(pos, neg, cfg, rng)
        logits, pred_len = net(x)
        loss_step = F.binary_cross_entropy_with_logits(logits, target)
        w = target * mask[:, None]                              # length only matters at stride peaks
        loss_len = (F.huber_loss(pred_len, length, reduction="none", delta=0.2) * w).sum() / w.sum().clamp(min=1.0)
        loss = loss_step + cfg.len_loss_weight * loss_len
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and (it % 100 == 0 or it == cfg.iters - 1):
            print(f"  iter {it:4d}  step-loss {loss_step.item():.4f}  len-loss {loss_len.item():.4f}")
    net.eval()
    return net


@torch.no_grad()
def predict(net, X):
    """(T,6) raw IMU -> (step probability (T,), stride length m (T,))."""
    x = torch.from_numpy(make_features(X)).T.unsqueeze(0)
    logits, length = net(x)
    return torch.sigmoid(logits)[0].numpy(), length[0].numpy()


def save_net(net, path):
    torch.save(net.state_dict(), path)


def load_net(path, cfg: StepConfig):
    net = StepNet(cfg)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net
