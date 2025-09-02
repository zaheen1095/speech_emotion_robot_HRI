# inspect_ckpts.py
import argparse
from pathlib import Path
import torch

def guess_input_dim(state_dict):
    # Look for first conv kernel weight: (out_ch, in_ch, kT)
    for k, v in state_dict.items():
        if k.endswith("conv.0.weight") and v.ndim == 3:
            return int(v.shape[1])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="models", help="Folder containing .pt checkpoints")
    args = ap.parse_args()

    p = Path(args.dir)
    pts = sorted(p.glob("*.pt"))
    if not pts:
        print(f"[warn] no .pt files under {p.resolve()}")
        return

    print("file, epoch, best_metric, sad_threshold, input_dim(D)")
    for f in pts:
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
                epoch = ckpt.get("epoch")
                best = ckpt.get("best_metric")
                thr  = ckpt.get("sad_threshold")
            else:
                sd, epoch, best, thr = ckpt, None, None, None
            D = guess_input_dim(sd)
            print(f"{f.name}, {epoch}, {best}, {thr}, {D}")
        except Exception as e:
            print(f"{f.name}, -, -, -, error: {e}")

if __name__ == "__main__":
    main()
