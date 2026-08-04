from pathlib import Path
import torch
import numpy as np
from src.model.model import MatchupClassifier

def get_nn_predictions(config: dict, X: np.ndarray) -> np.ndarray:
    '''loads the trained final checkpoint and runs inference on a feature matrix

    Parameters
    ----------
    config : dict
        config of the experiment, to locate the final checkpoint and build the model
    X : np.ndarray
        normalized feature matrix to predict on shape (N, F)

    Returns
    -------
    np.ndarray
        predicted probabilities shape (N,)
    '''
    experiment_name = config['model']['experiment_name']
    ckpt_path = Path(f'models/final/{experiment_name}/final.pt')

    model = MatchupClassifier(config)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)

    # no_grad since inference only
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)

    return probs.numpy().flatten()