import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix


def optimize_threshold_by_cost(y_true: np.ndarray, y_proba: np.ndarray,
                               fn_cost: float = 5.0, fp_cost: float = 1.0) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_proba).ravel()

    # Candidate thresholds from precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # precision_recall_curve returns thresholds of length n-1; add sentinel extremes
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))

    costs = []
    metrics = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn_cost * fn + fp_cost * fp
        pr_auc = average_precision_score(y_true, y_scores)
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        costs.append(cost)
        metrics.append({
            'threshold': float(t),
            'cost': float(cost),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'pr_auc': float(pr_auc),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        })

    best_idx = int(np.argmin(costs))
    return metrics[best_idx]
