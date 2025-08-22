# utils_threshold.py
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score

def sweep_threshold(y_true, p_sad, thr_min=0.30, thr_max=0.80, step=0.01, beta=1.0):
    """
    y_true : (N,) 0=happy, 1=sad
    p_sad  : (N,) probability of class 'sad'
    Returns: best_thr, stats dict (macro_f1, weighted_f1, f_beta, happy/sad recall, precision)
    """
    thrs = np.arange(thr_min, thr_max + 1e-9, step)
    best = {"thr": None, "macro_f1": -1, "weighted_f1": None, "f_beta": None,
            "happy_recall": None, "sad_recall": None, "precision": None}

    for thr in thrs:
        y_pred = (p_sad >= thr).astype(int)
        prec, rec, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0,1], zero_division=0
        )
        macro_f1 = np.mean(f1_per_class)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")

        if beta and beta != 1.0:
            # F_beta for the "sad" class (index 1)
            p1, r1 = prec[1], rec[1]
            if p1 + r1 > 0:
                f_beta = (1 + beta**2) * p1 * r1 / (beta**2 * p1 + r1 + 1e-12)
            else:
                f_beta = 0.0
        else:
            f_beta = macro_f1  # neutral choice when beta==1

        if macro_f1 > best["macro_f1"]:
            best.update({
                "thr": float(thr),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "f_beta": float(f_beta),
                "happy_recall": float(rec[0]),
                "sad_recall": float(rec[1]),
                "precision": float(np.mean(prec)),
            })
    return best["thr"], best
