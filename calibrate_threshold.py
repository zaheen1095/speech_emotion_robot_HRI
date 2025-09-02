# calibrate_temperature.py
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt

from models.cnn_bilstm import CNNBiLSTM
from config import FEATURES_DIR, MODEL_DIR, BATCH_SIZE, CLASSES, USE_ATTENTION
from sklearn.model_selection import GroupShuffleSplit

# ---- keep these aligned with your C3.1 script ----
AUG_TOKENS = (".aug","_aug","-aug","noise","reverb","rir","pitch","tempo","speed","stretch")

def _is_aug(name:str)->bool: return any(t in name.lower() for t in AUG_TOKENS)

def _label_from_path(p:Path)->int:
    parts = [s.lower() for s in p.parts]
    for ci, cname in enumerate(CLASSES):
        if cname.lower() in parts: return ci
    return CLASSES.index(p.parent.name)

def _scan_split(split:str, selected=None):
    root = Path(FEATURES_DIR)/split
    paths = list(root.rglob("*.npy"))
    labels = [_label_from_path(p) for p in paths]
    return paths, labels

def _infer_corpus(p:Path):
    # matches your Phase-B file naming convention (prefix before first "_")
    base = p.name.lower()
    return base.split("_",1)[0]

def _build_val_from_train(selected=None, val_fraction=0.20, seed=42):
    root = Path(FEATURES_DIR)/"train"
    paths, labels, groups = [], [], []
    for ci, cname in enumerate(CLASSES):
        for p in (root/cname).rglob("*.npy"):
            p=Path(p)
            paths.append(p); labels.append(ci)
            base = p.stem
            groups.append(f"{cname}/{base.split('__aug-')[0]}")
    if not paths: return [], []

    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr, va = next(gss.split(paths, labels, groups))
    Xv_all = [paths[i] for i in va]; yv_all = [labels[i] for i in va]
    Xv = [p for p in Xv_all if not _is_aug(p.name)]
    yv = [yv_all[i] for i,p in enumerate(Xv_all) if not _is_aug(p.name)]
    return Xv, yv

class FeatureDS(Dataset):
    def __init__(self, paths, labels):
        self.paths=[Path(p) for p in paths]; self.labels=labels
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x=np.load(self.paths[i]).astype(np.float32)
        y=int(self.labels[i])
        return torch.tensor(x), torch.tensor(y)

def _pad_collate(batch):
    xs, ys = zip(*batch)
    T = max(x.shape[0] for x in xs); D = xs[0].shape[1]
    out = torch.zeros(len(xs), T, D, dtype=torch.float32)
    for i,x in enumerate(xs): out[i,:x.shape[0]] = x
    return out, torch.stack(ys)

def _load_model(input_dim, device, ckpt_path):
    m = CNNBiLSTM(input_dim=input_dim, num_classes=len(CLASSES), use_attention=USE_ATTENTION).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    m.load_state_dict(state, strict=True)
    m.eval()
    return m

@torch.no_grad()
def _logits_labels(model, loader, device):
    logits, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        lg = model(xb).cpu()
        logits.append(lg); labels.extend(yb.numpy().tolist())
    return torch.cat(logits,0), np.array(labels)

def _ece(probs, y_true, n_bins=15):
    # probs: [N,C], y_true: [N]
    conf = probs.max(1)                     # confidence
    preds = probs.argmax(1)
    acc   = (preds==y_true).astype(float)
    bins = np.linspace(0,1,n_bins+1)
    ece=0.0
    for i in range(n_bins):
        lo,hi = bins[i], bins[i+1]
        sel = (conf>=lo)&(conf<hi)
        if not sel.any(): continue
        gap = abs(acc[sel].mean() - conf[sel].mean())
        ece += (sel.mean()) * gap
    return float(ece)

def _reliability_diagram(probs, y_true, out_png, n_bins=15):
    conf = probs.max(1)
    preds = probs.argmax(1)
    acc   = (preds==y_true).astype(float)
    bins = np.linspace(0,1,n_bins+1)
    mids, accs, confs = [], [], []
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        sel=(conf>=lo)&(conf<hi)
        if sel.any():
            mids.append((lo+hi)/2)
            accs.append(acc[sel].mean())
            confs.append(conf[sel].mean())
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4), dpi=140)
    plt.plot([0,1],[0,1],'--',linewidth=1)
    plt.plot(confs, accs, marker='o')
    plt.xlabel('Confidence'); plt.ylabel('Accuracy'); plt.title('Reliability')
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(Path(MODEL_DIR)/"best_model.pt"))
    ap.add_argument("--out_dir",   type=str, default="results/C3_temp_calib")
    args = ap.parse_args()

    # Build val from train (no need for FEATURES_DIR/val on disk)
    Xv, yv = _build_val_from_train()
    Xt, yt = _scan_split("test")

    assert Xv, "No validation features found/constructed."
    assert Xt, "No test features found."

    # infer input dim
    d = np.load(Xv[0]).shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_model(d, device, args.checkpoint)

    val_loader  = DataLoader(FeatureDS(Xv, yv), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True, collate_fn=_pad_collate)
    test_loader = DataLoader(FeatureDS(Xt, yt), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True, collate_fn=_pad_collate)

    # ---- collect logits ----
    Lv, yv = _logits_labels(model, val_loader, device)
    Lt, yt = _logits_labels(model, test_loader, device)

    # ---- fit temperature on VAL (minimize NLL) ----
    T = torch.nn.Parameter(torch.ones(1, device=Lv.device))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    # cache labels once on the correct device/dtype
    yv_t = torch.tensor(yv, dtype=torch.long, device=Lv.device)

    def _closure():
        opt.zero_grad(set_to_none=True)
        scaled = Lv / T.clamp(min=1e-6)
        loss = torch.nn.functional.cross_entropy(scaled, yv_t)
        loss.backward()
        return loss

    opt.step(_closure)
    T_star = float(T.detach().cpu().item())

    # ---- evaluate on TEST before/after ----
    p_test_uncal = F.softmax(Lt, dim=1).numpy()
    p_test_cal   = F.softmax(Lt / T_star, dim=1).numpy()

    nll_uncal = log_loss(yt, p_test_uncal, labels=list(range(len(CLASSES))))
    nll_cal   = log_loss(yt, p_test_cal,   labels=list(range(len(CLASSES))))
    brier_uncal = brier_score_loss((np.array(yt)==1).astype(int), p_test_uncal[:,1])
    brier_cal   = brier_score_loss((np.array(yt)==1).astype(int), p_test_cal[:,1])
    ece_uncal = _ece(p_test_uncal, np.array(yt))
    ece_cal   = _ece(p_test_cal,   np.array(yt))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # reliability diagrams
    _reliability_diagram(p_test_uncal, np.array(yt), out/"reliability_uncal.png")
    _reliability_diagram(p_test_cal,   np.array(yt), out/"reliability_cal.png")

    with open(out/"temperature.json","w") as f:
        json.dump({
            "T": T_star,
            "val_samples": len(yv),
            "metrics_test": {
                "nll_uncal": nll_uncal, "nll_cal": nll_cal,
                "brier_uncal": brier_uncal, "brier_cal": brier_cal,
                "ece_uncal": ece_uncal, "ece_cal": ece_cal
            }
        }, f, indent=2)

    print(f"[C3.2] Learned temperature T={T_star:.3f}")
    print(f"[C3.2] Test NLL: {nll_uncal:.4f} → {nll_cal:.4f}")
    print(f"[C3.2] Brier:    {brier_uncal:.4f} → {brier_cal:.4f}")
    print(f"[C3.2] ECE:      {ece_uncal:.4f} → {ece_cal:.4f}")
    print(f"[C3.2] Wrote {out/'temperature.json'} and reliability diagrams.")

if __name__ == "__main__":
    main()
