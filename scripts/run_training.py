'''
orchestrates the individual model training process to experiment on individual models

checkpoints based on config and saves bet val loss to models/checkpoints/model_experiment_name 
and models/final/model_experiment_name

change experiment by editing CONFIG_PATH constant to match config location relative to root

run using python -m scripts.run_training.py
'''

from src.utils.config import load_config
from src.model.model import MatchupClassifier
from src.training.train import train_model
import numpy as np

CONFIG_PATH = 'configs/upset.yaml'

def main():
    config = load_config(CONFIG_PATH)

    experiment_name = config['data']['experiment_name']
    splits_dir = f'{config["data"]["splits_dir"]}/{experiment_name}'

    X_train = np.load(f'{splits_dir}/X_train.npy')
    y_train = np.load(f'{splits_dir}/y_train.npy')
    X_val = np.load(f'{splits_dir}/X_val.npy')
    y_val = np.load(f'{splits_dir}/y_val.npy')

    model = MatchupClassifier(config)

    train_model(config, model, X_train, y_train, X_val, y_val)

if __name__ == '__main__':
    main()