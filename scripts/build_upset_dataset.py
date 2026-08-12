'''
builds the dataset for the upset prediction data. separate script b/c upset data requires different split format
enforcing winner first format so that label/model output can indicate whether an upset happened. outputs team stats csv 
with per season per team agg stats in data/processed/team_stats_data_experiment_name.csv and the 
final train/val/test splits in data/splits/data_experiment_name

change data experiment by editing CONFIG_PATH constant based on data experiement

run using python -m scripts.build_upset_dataset --config configs/upset.yaml
'''


import argparse
from src.data.upset_dataset import load_upset_data, build_upset_splits
from src.data.preprocess import build_team_stats
from src.data.ingest import load_raw_data
from src.utils.config import load_config

CONFIG_PATH = 'configs/upset.yaml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=CONFIG_PATH, help='path to config yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    data = load_raw_data(config['data']['raw_dir'])
    raw_season_results = data['season_results'].copy()
    raw_tourney_results = data['tourney_results'].copy()
    raw_seeds = data['seeds'].copy()

    build_team_stats(config, raw_season_results)

    team_stats, seeds, tourney_results = load_upset_data(config, raw_tourney_results, raw_seeds)
    build_upset_splits(config, team_stats, seeds, tourney_results)