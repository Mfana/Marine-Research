"""Spatiotemporal analysis and deep-learning field forecasting (paper 3).

- EOF/PCA of area-weighted anomaly fields with North et al. (1982) sampling errors.
- Spatial decorrelation length scales.
- ConvLSTM (Shi et al. 2015) and a small U-Net-style CNN forecasting anomaly fields,
  trained with a sea-masked MSE loss.
"""

from __future__ import annotations

import numpy as np


# ---- EOF ---------------------------------------------------------------------

def eof(anom: np.ndarray, lat: np.ndarray, n_modes: int = 4) -> dict:
    """anom: (time, lat, lon) with NaN on land. Returns patterns (in °C per unit PC),
    standardised PCs, explained variance fractions and North's rule-of-thumb errors."""
    t, ny, nx = anom.shape
    w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * np.ones(nx)
    sea = ~np.isnan(anom).any(0)
    X = (anom * w)[:, sea]
    X = X - X.mean(0)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    var = s ** 2
    frac = var / var.sum()
    pcs = u[:, :n_modes] * s[:n_modes]
    pcs_std = pcs / pcs.std(0)
    patterns = np.full((n_modes, ny, nx), np.nan)
    for k in range(n_modes):
        # regression of the (unweighted) anomaly onto the standardised PC: °C per 1 s.d.
        patterns[k][sea] = (anom[:, sea] - np.nanmean(anom[:, sea], 0)).T @ pcs_std[:, k] / t
    n_eff = t  # conservative choice is documented by the caller (autocorrelation reduces it)
    north_err = frac[:n_modes + 1] * np.sqrt(2.0 / n_eff)
    return {"patterns": patterns, "pcs": pcs_std, "frac": frac[:n_modes], "north_err": north_err[:n_modes],
            "frac_next": frac[n_modes] if len(frac) > n_modes else np.nan}


def effective_sample_size(x: np.ndarray) -> float:
    """n_eff = n (1 - r1) / (1 + r1) for a lag-1 autocorrelated series."""
    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    return len(x) * (1 - r1) / (1 + r1)


def decorrelation_length(anom: np.ndarray, lat: np.ndarray, lon: np.ndarray, points: list[tuple[int, int]],
                         max_km: float = 600, bin_km: float = 25) -> dict:
    """For each reference cell (iy, ix), correlation of its anomaly series with all sea
    cells, binned by distance; e-folding distance where the binned mean falls below 1/e."""
    ny, nx = len(lat), len(lon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    sea = ~np.isnan(anom).any(0)
    flat = anom.reshape(anom.shape[0], -1)
    fz = (flat - np.nanmean(flat, 0)) / np.nanstd(flat, 0)
    out = {}
    for iy, ix in points:
        ref = fz[:, iy * nx + ix]
        r = np.nanmean(fz * ref[:, None], 0).reshape(ny, nx)
        dy = (LAT - lat[iy]) * 110.57
        dx = (LON - lon[ix]) * 111.32 * np.cos(np.deg2rad(lat[iy]))
        d = np.hypot(dx, dy)
        edges = np.arange(0, max_km + bin_km, bin_km)
        centers, means = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = sea & (d >= a) & (d < b)
            if m.sum() >= 5:
                centers.append((a + b) / 2)
                means.append(float(np.nanmean(r[m])))
        centers, means = np.array(centers), np.array(means)
        # e-folding distance: linear interpolation where the binned mean first drops below 1/e.
        below = np.where(means < np.exp(-1))[0]
        if not len(below):
            L = np.nan                      # still correlated beyond max_km
        elif below[0] == 0:
            L = float(centers[0])           # decorrelates within the first bin
        else:
            j = below[0]
            m0, m1, c0, c1 = means[j - 1], means[j], centers[j - 1], centers[j]
            L = float(c0 + (m0 - np.exp(-1)) / (m0 - m1) * (c1 - c0))
        out[(iy, ix)] = {"distance_km": centers, "mean_r": means, "efold_km": L}
    return out


# ---- deep learning -------------------------------------------------------------

def build_models():
    import torch
    import torch.nn as nn

    class ConvLSTMCell(nn.Module):
        def __init__(self, c_in, c_hid, k=3):
            super().__init__()
            self.c_hid = c_hid
            self.conv = nn.Conv2d(c_in + c_hid, 4 * c_hid, k, padding=k // 2)

        def forward(self, x, state):
            h, c = state
            i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            return h, c

    class ConvLSTMNet(nn.Module):
        """Encoder ConvLSTM over the input weeks; 1×1 head maps the last hidden state
        to one anomaly field per forecast lead."""

        def __init__(self, c_in, n_out, hidden=(32, 32)):
            super().__init__()
            self.cells = nn.ModuleList()
            prev = c_in
            for h in hidden:
                self.cells.append(ConvLSTMCell(prev, h))
                prev = h
            self.head = nn.Conv2d(prev, n_out, 1)

        def forward(self, x):            # x: (B, T, C, H, W)
            B, T, _, H, W = x.shape
            states = [(x.new_zeros(B, c.c_hid, H, W), x.new_zeros(B, c.c_hid, H, W)) for c in self.cells]
            for t in range(T):
                inp = x[:, t]
                for j, cell in enumerate(self.cells):
                    states[j] = cell(inp, states[j])
                    inp = states[j][0]
            return self.head(inp)

    class UNetLite(nn.Module):
        """Input weeks stacked as channels; two-level encoder–decoder with skip connections."""

        def __init__(self, c_in, n_out, base=32):
            super().__init__()
            blk = lambda a, b: nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GELU(),
                                             nn.Conv2d(b, b, 3, padding=1), nn.GELU())
            self.e1, self.e2 = blk(c_in, base), blk(base, 2 * base)
            self.pool = nn.MaxPool2d(2)
            self.mid = blk(2 * base, 2 * base)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.d2, self.d1 = blk(4 * base, base), blk(2 * base, base)
            self.head = nn.Conv2d(base, n_out, 1)

        def forward(self, x):            # x: (B, T, C, H, W) -> stack T*C as channels
            B, T, C, H, W = x.shape
            x = x.reshape(B, T * C, H, W)
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            m = self.mid(e2)
            d2 = self.d2(torch.cat([m, e2], 1))
            d1 = self.d1(torch.cat([self.up(d2), e1], 1))
            return self.head(d1)

    return ConvLSTMNet, UNetLite


def train_field_model(model, Xtr, Ytr, Xva, Yva, mask, epochs=200, lr=1e-3, patience=20,
                      batch=32, seed=0, device="cpu", log=None):
    """Masked-MSE training with early stopping on validation loss. Arrays are numpy:
    X (N, T, C, H, W), Y (N, L, H, W); mask (H, W) True on sea. With Xva=None the model
    trains for exactly `epochs` epochs and keeps the final weights (refit mode)."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = torch.tensor(mask, dtype=torch.float32, device=device)
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    xtr, ytr = to(Xtr), to(Ytr)
    refit = Xva is None
    if not refit:
        xva, yva = to(Xva), to(Yva)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = lambda p, y: ((p - y) ** 2 * m).sum() / (m.sum() * p.shape[0] * p.shape[1])
    best, best_state, wait, best_epoch = np.inf, None, 0, 0
    import time
    t_start = time.time()
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=gen).to(device)
        for i in range(0, len(xtr), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        if log and (epoch + 1) % 10 == 0:
            log.info("    epoch %d (%.1fs/epoch)", epoch + 1, (time.time() - t_start) / (epoch + 1))
        if refit:
            continue
        model.eval()
        with torch.no_grad():
            v = float(sum(loss_fn(model(xva[i:i + 64]), yva[i:i + 64]) * len(xva[i:i + 64])
                          for i in range(0, len(xva), 64)) / len(xva))
        if v < best - 1e-6:
            best, wait, best_epoch = v, 0, epoch + 1
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if refit:
        return model, np.nan, epochs
    model.load_state_dict(best_state)
    if log:
        log.info("  trained %s: best val %.4f at epoch %d", type(model).__name__, best, best_epoch)
    return model, best, best_epoch


def predict_fields(model, X, device="cpu", batch=64):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)).cpu().numpy())
    return np.concatenate(out)
