import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

def train_model(config, model, X_train, y_train, X_val, y_val):
    train_config = config['training']
    log_config = config['logging']

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

    loss_fn = nn.BCEWithLogitsLoss()

    ckpt_dir = Path('models/checkpoints') / log_config['experiment_name']
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_dir = Path('models/final') / log_config['experiment_name']
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / 'final.pt'

    best_val_loss = float('inf')
    checkpoint_every = log_config['checkpoint_every_n_epochs']

    history = {'train_loss': [], 'val_loss': []}

    pbar = tqdm(range(1, train_config['epochs'] + 1), desc='training model', position=0, leave=True, unit='epoch')
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
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f'val epoch {epoch:02d}', leave=False, position=1, unit='batch'):
                logits = model(xb)
                loss = loss_fn(logits, yb)
                running_val_loss += loss.item() * xb.size(0)
        val_loss = running_val_loss / len(val_ds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        pbar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), final_path)

        if epoch % checkpoint_every == 0:
            periodic_path = ckpt_dir / f'epoch_{epoch}.pt'
            torch.save(model.state_dict(), periodic_path)

    