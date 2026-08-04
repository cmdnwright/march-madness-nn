import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.training.losses import FocalLoss

def train_model(config: dict, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
    '''trains the model, checkpointing periodically and saving the best val loss model as final

    Parameters
    ----------
    config : dict
        config of the experiment for training hyperparams and save locations
    model : nn.Module
        model to train
    X_train : np.ndarray
        train features shape (N, F)
    y_train : np.ndarray
        train labels shape (N,)
    X_val : np.ndarray
        val features shape (N, F)
    y_val : np.ndarray
        val labels shape (N,)
    '''
    train_config = config['training']
    model_config = config['model']

    seed = train_config['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )

    val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32), 
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )

    train_loader = DataLoader(train_ds, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_config['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])

    loss_fn = FocalLoss(gamma=train_config['focal_gamma'])

    ckpt_dir = Path('models/checkpoints') / model_config['experiment_name']
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_dir = Path('models/final') / model_config['experiment_name']
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / 'final.pt'

    best_val_loss = float('inf')
    checkpoint_every = train_config['checkpoint_every_n_epochs']

    history = {'train_loss': [], 'val_loss': []}

    pbar = tqdm(range(1, train_config['epochs'] + 1), desc='training model', leave=False, unit='epoch')
    for epoch in pbar:
        model.train()
        running_train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f'train epoch {epoch:02d}', leave=False, position=1, unit='batch'):
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * xb.size(0)
        train_loss = running_train_loss / len(train_ds)

        model.eval()
        running_val_loss = 0.0
        # no_grad since validation
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f'val epoch {epoch:02d}', leave=False, position=1, unit='batch'):
                logits = model(xb)
                loss = loss_fn(logits, yb)
                running_val_loss += loss.item() * xb.size(0)
        val_loss = running_val_loss / len(val_ds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        pbar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})

        # always keep the final checkpoint as the best val loss seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), final_path)

        if epoch % checkpoint_every == 0:
            periodic_path = ckpt_dir / f'epoch_{epoch}.pt'
            torch.save(model.state_dict(), periodic_path)