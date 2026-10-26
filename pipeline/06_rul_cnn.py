"""Stage 6 - deep learning RUL model: a 1D convolutional network over 30-cycle sensor windows,
trained on the Apple-silicon GPU (PyTorch MPS backend; falls back to CUDA or CPU).

Windows are gathered on the device from one padded tensor (no 30x copy of the data in RAM).
Uses the exact engine-grouped folds of stage 5 so the two models' out-of-fold predictions
are comparable and can be stacked.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from torch import nn

from importlib import import_module

from common import PROCESSED, SEED, Timer, export

gbm = import_module("05_rul_gbm")   # reuse the exact feature builder of stage 5

WINDOW = 30
EPOCHS = 18
BATCH = 512
INFORMATIVE = gbm.INFORMATIVE
torch.manual_seed(SEED)
np.random.seed(SEED)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Windows:
    """Left-pad each engine with copies of its first row; window i ends at ends[i]."""

    def __init__(self, df: pd.DataFrame, chans: list[str], dev: torch.device):
        blocks, ends, pos = [], [], 0
        for _, g in df.groupby("engine_id", sort=False):
            x = g[chans].to_numpy(np.float32)
            pad = np.repeat(x[:1], WINDOW - 1, axis=0)
            blocks.append(np.vstack([pad, x]))
            ends.append(pos + WINDOW - 1 + np.arange(len(x)))
            pos += len(x) + WINDOW - 1
        self.data = torch.from_numpy(np.vstack(blocks)).to(dev)
        self.ends = torch.from_numpy(np.concatenate(ends)).to(dev)
        self.offsets = torch.arange(-(WINDOW - 1), 1, device=dev)

    def get(self, idx: torch.Tensor) -> torch.Tensor:
        pos = self.ends[idx][:, None] + self.offsets[None, :]
        return self.data[pos].transpose(1, 2)  # (B, C, W)


class RULNet(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, 64, 5, padding=2), nn.GELU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 5, padding=2), nn.GELU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 48, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(48 * 2 + c_in, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        h = self.conv(x)
        z = torch.cat([h.mean(-1), h[..., -1], x[..., -1]], dim=1)  # pooled + latest state + raw latest
        return self.head(z).squeeze(-1)


def train_one(win: Windows, y: torch.Tensor, idx_tr: np.ndarray, dev: torch.device, epochs=EPOCHS, log=False):
    model = RULNet(win.data.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (len(idx_tr) // BATCH + 1))
    idx_tr = torch.from_numpy(idx_tr).to(dev)
    times = []
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        perm = idx_tr[torch.randperm(len(idx_tr), device=dev)]
        total = 0.0
        for b in range(0, len(perm), BATCH):
            bi = perm[b:b + BATCH]
            pred = model(win.get(bi))
            loss = nn.functional.mse_loss(pred, y[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(bi)
        if dev.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        if log:
            print(f"    epoch {ep + 1:2d}  train RMSE {np.sqrt(total / len(perm)):.2f}  ({times[-1]:.1f}s)")
    return model, times


@torch.no_grad()
def predict(model, win: Windows, n: int, dev) -> np.ndarray:
    model.eval()
    out = []
    for b in range(0, n, 4096):
        idx = torch.arange(b, min(b + 4096, n), device=dev)
        out.append(model(win.get(idx)).float().cpu().numpy())
    return np.concatenate(out)


def benchmark_epoch(df_tr: pd.DataFrame, chans: list[str], target: np.ndarray) -> dict:
    """Same one-epoch workload on CPU vs GPU, for an honest speed comparison."""
    res = {}
    for name in ("cpu", device().type):
        dev = torch.device(name)
        win = Windows(df_tr, chans, dev)
        y = torch.from_numpy(target).to(dev)
        _, times = train_one(win, y, np.arange(len(df_tr)), dev, epochs=2)
        res[name] = times[-1]
    return res


def main() -> None:
    dev = device()
    print(f"  device: {dev}")
    df = gbm.build_features()
    chans = [f"{s}_n" for s in INFORMATIVE] + [f"{s}_dev" for s in INFORMATIVE] + ["cycle_scaled"]
    df["cycle_scaled"] = df.cycle / 200.0
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    oof_gbm = pd.read_parquet(PROCESSED / "rul_oof_gbm.parquet")
    assert (oof_gbm.engine_id.values == tr.engine_id.values).all() and (oof_gbm.cycle.values == tr.cycle.values).all()
    fold = oof_gbm.fold.to_numpy()

    with Timer("CPU vs GPU benchmark") as tb:
        bench = benchmark_epoch(tr, chans, tr.target.to_numpy(np.float32))
    print(f"  one epoch: {bench}")

    win_tr = Windows(tr, chans, dev)
    win_te = Windows(te, chans, dev)
    y = torch.from_numpy(tr.target.to_numpy(np.float32)).to(dev)
    oof = np.zeros(len(tr), np.float32)
    test_preds = []
    with Timer("5-fold CNN training") as tt:
        for k in range(fold.max() + 1):
            i_tr, i_va = np.where(fold != k)[0], np.where(fold == k)[0]
            model, _ = train_one(win_tr, y, i_tr, dev, log=(k == 0))
            p_all = predict(model, win_tr, len(tr), dev)
            oof[i_va] = p_all[i_va]
            test_preds.append(predict(model, win_te, len(te), dev))
            print(f"  fold {k + 1}: CNN OOF RMSE {np.sqrt(np.mean((oof[i_va] - tr.target.values[i_va]) ** 2)):.2f}")
    torch.save(model.state_dict(), PROCESSED / "cnn_rul_fold5.pt")

    out_tr = tr[["engine_id", "cycle"]].copy()
    out_tr["oof_cnn"] = oof
    out_te = te[["engine_id", "cycle"]].copy()
    out_te["pred_cnn"] = np.mean(test_preds, axis=0)
    out_tr.to_parquet(PROCESSED / "rul_oof_cnn.parquet")
    out_te.to_parquet(PROCESSED / "rul_test_cnn.parquet")

    last = te.assign(p=out_te.pred_cnn).groupby("engine_id").tail(1)
    e = last.p.clip(0, None) - last.rul
    print(f"  TEST cnn RMSE {np.sqrt(np.mean(e ** 2)):.2f}  " +
          "  ".join(f"{s}:{np.sqrt(np.mean(e[last.subset == s] ** 2)):.2f}" for s in sorted(last.subset.unique())))
    n_params = sum(p.numel() for p in RULNet(len(chans)).parameters())
    export("gpu", {"device": dev.type, "torch": torch.__version__, "epoch_seconds": bench, "params": n_params,
                   "window": WINDOW, "channels": len(chans), "epochs": EPOCHS, "folds": int(fold.max() + 1),
                   "train_seconds": tt.seconds, "train_windows": len(tr)})


if __name__ == "__main__":
    main()
