from pathlib import Path
import torch
import numpy as np
from src.model.model import MatchupClassifier

def get_nn_predictions(config, X):
    experiment_name = config['model']['experiment_name']
    ckpt_path = Path(f'models/final/{experiment_name}/final.pt')

    model = MatchupClassifier(config)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)

    return probs.numpy().flatten()