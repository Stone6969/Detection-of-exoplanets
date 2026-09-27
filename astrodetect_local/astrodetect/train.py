"""Training + evaluation (run on a Kaggle GPU).

  load shards -> pick detrending variant (raw = ours, pdc = NASA baseline)
  -> train AstroNetPlus with the partial-label loss (class-balanced sampling)
  -> early stopping on validation loss -> temperature calibration
  -> test metrics on UNSEEN stars: real planet-vs-FP ROC-AUC / PR-AUC,
     4-class report on simulated test samples, and the raw-vs-pdc ablation.
"""
import glob
import json
import os

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)

from .model import PARTIAL_NOT_PLANET, AstroNetPlus, Standardizer, augment, partial_label_loss
from .significance import TemperatureScaler
from .simulate import CLASSES


def load_shards(pattern):
    parts = [dict(np.load(f, allow_pickle=False)) for f in sorted(glob.glob(pattern, recursive=True))]
    if not parts:
        raise FileNotFoundError(pattern)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def subset(D, m):
    return {k: v[m] for k, v in D.items()}


def select(D, variant, split, include_unlabeled=False):
    m = (D["variant"] == variant) & (D["split"] == split)
    if not include_unlabeled:
        m &= D["y"] != -1
    return subset(D, m)


class DS(torch.utils.data.Dataset):
    def __init__(self, D, scaler, train):
        self.G, self.L, self.S, self.y = D["G"], D["L"], scaler(D["S"]), D["y"]
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        g, l = self.G[i], self.L[i]
        if self.train:
            g, l = augment(g, l)
        return (torch.from_numpy(np.ascontiguousarray(g, np.float32)),
                torch.from_numpy(np.ascontiguousarray(l, np.float32)),
                torch.from_numpy(self.S[i]), int(self.y[i]))


def _weights(y):
    """Balanced sampling over {sim planet, sim EB, sim blend, sim other,
    real planet, real FP}."""
    keys = np.where(y == PARTIAL_NOT_PLANET, 10, y)
    u, c = np.unique(keys, return_counts=True)
    w = {k: 1.0 / n for k, n in zip(u, c)}
    return np.array([w[k] for k in keys])


@torch.no_grad()
def predict_logits(model, D, scaler, device, bs=512):
    model.eval()
    dl = torch.utils.data.DataLoader(DS(D, scaler, False), batch_size=bs)
    out = [model(g.to(device), l.to(device), s.to(device)).cpu().numpy() for g, l, s, _ in dl]
    return np.concatenate(out) if out else np.zeros((0, 4))


def train_variant(D, variant, out_dir, epochs=40, bs=128, lr=1e-3, patience=6, seed=0,
                  device=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = (select(D, variant, s) for s in ("train", "val", "test"))
    scaler = Standardizer().fit(tr["S"])
    sampler = torch.utils.data.WeightedRandomSampler(_weights(tr["y"]), len(tr["y"]), replacement=True)
    dl = torch.utils.data.DataLoader(DS(tr, scaler, True), batch_size=bs, sampler=sampler,
                                     num_workers=2, drop_last=True)
    model = AstroNetPlus().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=epochs, steps_per_epoch=len(dl))
    best, bad, hist = np.inf, 0, []
    yv = torch.tensor(va["y"])
    for ep in range(epochs):
        model.train()
        tl = []
        for g, l, s, y in dl:
            loss = partial_label_loss(model(g.to(device), l.to(device), s.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl.append(loss.item())
        vl = partial_label_loss(torch.tensor(predict_logits(model, va, scaler, device)), yv).item()
        hist.append(dict(epoch=ep, train_loss=float(np.mean(tl)), val_loss=vl))
        print(f"[{variant}] epoch {ep:02d} train {np.mean(tl):.4f} val {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save(model.state_dict(), os.path.join(out_dir, f"model_{variant}.pt"))
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(torch.load(os.path.join(out_dir, f"model_{variant}.pt"), map_location=device))

    # temperature calibration on fully-labelled validation samples
    lv = predict_logits(model, va, scaler, device)
    full = va["y"] >= 0
    temp = TemperatureScaler().fit(lv[full], va["y"][full]) if full.sum() > 20 else TemperatureScaler()
    json.dump(dict(scaler=scaler.state(), temperature=temp.T, history=hist),
              open(os.path.join(out_dir, f"model_{variant}.json"), "w"))
    metrics = evaluate(model, te, scaler, temp, device)
    json.dump(metrics, open(os.path.join(out_dir, f"metrics_{variant}.json"), "w"), indent=2)
    return model, scaler, temp, metrics


def evaluate(model, te, scaler, temp, device):
    P = temp(predict_logits(model, te, scaler, device))
    m = {}
    real = te["kind"] == "real"
    rl = real & (te["y"] != -1)
    if rl.sum() and len(np.unique(te["y"][rl] == 0)) == 2:
        yb = (te["y"][rl] == 0).astype(int)
        m["real_planet_vs_fp_roc_auc"] = float(roc_auc_score(yb, P[rl, 0]))
        m["real_planet_vs_fp_pr_auc"] = float(average_precision_score(yb, P[rl, 0]))
        m["real_n"] = int(rl.sum())
        m["real_n_planets"] = int(yb.sum())
        # recall of real planets / FP rejection at p_planet >= 0.5
        pred = P[rl, 0] >= 0.5
        m["real_planet_recall@0.5"] = float(pred[yb == 1].mean())
        m["real_fp_rejected@0.5"] = float((~pred[yb == 0]).mean())
    sim = te["kind"] == "sim"
    if sim.sum():
        yt, yp = te["y"][sim], P[sim].argmax(1)
        m["sim_4class_accuracy"] = float((yt == yp).mean())
        m["sim_confusion"] = confusion_matrix(yt, yp, labels=range(4)).tolist()
        m["sim_report"] = classification_report(yt, yp, labels=range(4), target_names=CLASSES,
                                                output_dict=True, zero_division=0)
    return m


def load_model(out_dir, variant="raw", device="cpu"):
    cfg = json.load(open(os.path.join(out_dir, f"model_{variant}.json")))
    model = AstroNetPlus()
    model.load_state_dict(torch.load(os.path.join(out_dir, f"model_{variant}.pt"), map_location=device))
    model.eval()
    scaler = Standardizer.from_state(cfg["scaler"])
    temp = TemperatureScaler(cfg["temperature"])

    @torch.no_grad()
    def predict(g, l, s):
        z = model(torch.from_numpy(g[None]).float(), torch.from_numpy(l[None]).float(),
                  torch.from_numpy(scaler(s[None]))).numpy()
        return temp(z)[0]
    return predict
