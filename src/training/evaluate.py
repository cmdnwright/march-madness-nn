import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


def evaluate(probs: np.ndarray, labels: np.ndarray) -> dict:
    '''computes standard binary classification metrics from predicted probabilities and true labels

    Parameters
    ----------
    probs : np.ndarray
        predicted probabilities shape (N,)
    labels : np.ndarray
        true binary labels shape (N,)

    Returns
    -------
    dict
        accuracy, auc, log loss, and brier score of the predictions
    '''
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()

    preds = (probs > 0.5).astype(int)
    accuracy = float(np.mean(preds == labels))
    auc = roc_auc_score(labels, probs)
    loss = log_loss(labels, probs)
    brier = brier_score_loss(labels, probs)

    return {
        'accuracy': accuracy,
        'auc': auc,
        'loss': loss,
        'brier': brier
    }