import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def evaluate(probs, labels):
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()

    preds = (probs > 0.5).astype(int)
    accuracy = float(np.mean(preds == labels))
    auc = roc_auc_score(labels, probs)
    loss = log_loss(labels, probs)

    return {
        'accuracy': accuracy,
        'auc': auc,
        'loss': loss,
    }