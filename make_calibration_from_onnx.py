#make_calibration_from_onnx.py

import argparse, json, os, numpy as np, onnxruntime as ort, pathlib

def softmax(x, T=1.0):
    x = x / T
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def load_split(features_root, split="val"):
    X, y = [], []
    # expects e.g. features_root/val/<class>/*.npy where class in {"happy","sad"}
    for cls, label in [("happy",0),("sad",1)]:
        p = pathlib.Path(features_root)/split/cls
        if not p.exists(): 
            continue
        for f in p.rglob("*.npy"):
            X.append(np.load(f))
            y.append(label)
    if not X:
        raise FileNotFoundError(f"No .npy features under {features_root}/{split}/happy|sad")
    # pad/truncate to same T
    T = max(x.shape[0] for x in X)
    D = X[0].shape[1]
    Xp = np.zeros((len(X), T, D), np.float32)
    for i,x in enumerate(X):
        t = min(T, x.shape[0])
        Xp[i,:t,:] = x[:t,:]
    return Xp, np.array(y, dtype=np.int64)

def macro_f1(y_true, y_pred):
    f1s = []
    for c in [0,1]:
        tp = np.sum((y_true==c)&(y_pred==c))
        fp = np.sum((y_true!=c)&(y_pred==c))
        fn = np.sum((y_true==c)&(y_pred!=c))
        prec = tp/(tp+fp+1e-9)
        rec  = tp/(tp+fn+1e-9)
        f1 = 2*prec*rec/(prec+rec+1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))

def nll_loss(p, y):
    # p shape (N,2); y in {0,1}
    p = np.clip(p, 1e-7, 1-1e-7)
    return -float(np.mean(np.log(p[np.arange(len(y)), y])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--features_root", required=True)  # e.g. datasets/features_ssl
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)            # e.g. models/ssl_v1/calibration.json
    ap.add_argument("--no_temperature", action="store_true")
    args = ap.parse_args()

    X, y = load_split(args.features_root, args.split)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    # forward logits
    logits = []
    B = 32
    for i in range(0, len(X), B):
        batch = X[i:i+B]
        out = sess.run(None, {inp_name: batch})[0]   # (B,2)
        logits.append(out)
    logits = np.concatenate(logits, axis=0)          # (N,2)

    # grid search temperature (SSL) to minimize NLL
    temps = [1.0]
    if not args.no_temperature:
        temps = np.round(np.linspace(0.8, 2.0, 25), 2).tolist()

    best = {"temperature": 1.0, "threshold": 0.5, "macro_f1": -1, "nll": 1e9}
    for T in temps:
        probs = softmax(logits, T=T)
        # sweep thresholds for 'sad' class (index 1)
        psad = probs[:,1]
        for t in np.round(np.linspace(0.30, 0.70, 81), 3):
            pred = (psad >= t).astype(int)  # 1=sad, 0=happy
            f1 = macro_f1(y, pred)
            nll = nll_loss(probs, y)
            if f1 > best["macro_f1"] or (abs(f1-best["macro_f1"])<1e-6 and nll<best["nll"]):
                best = {"temperature": float(T), "threshold": float(t), "macro_f1": f1, "nll": nll}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {"threshold": best["threshold"]}
    if not args.no_temperature:
        out["temperature"] = best["temperature"]
    out["calibrated_on"] = args.split
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", args.out, "→", out, "| macroF1(val)=", round(best["macro_f1"],4), "NLL=", round(best["nll"],4))

if __name__ == "__main__":
    main()
