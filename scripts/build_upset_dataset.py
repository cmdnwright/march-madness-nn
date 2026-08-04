'''
builds the dataset for the upset prediction data. separate script b/c upset data requires different split format
enforcing winner first format so that label/model output can indicate whether an upset happened. outputs team stats csv 
with per season per team agg stats in data/processed/team_stats_data_experiment_name.csv and the 
final train/val/test splits in data/splits/data_experiment_name

change data experiment by editing CONFIG_PATH constant based on data experiement

run using python -m scripts.build_upset_dataset.py
'''


from src.data.upset_dataset import load_upset_data, build_upset_splits
from src.data.preprocess import build_team_stats
from src.data.ingest import load_raw_data
from src.utils.config import load_config

CONFIG_PATH = 'configs/upset.yaml'

if __name__ == '__main__':
    config = load_config(CONFIG_PATH)

    data = load_raw_data(config['data']['raw_dir'])
    raw_season_results = data['season_results'].copy()
    raw_tourney_results = data['tourney_results'].copy()
    raw_seeds = data['seeds'].copy()

    # upset.yaml has its own data.experiment_name ('upset'), so this writes
    # team_stats_upset.csv -- a duplicate of team_stats_baseline.csv in
    # content, since the box-score aggregation is identical, but keeping it
    # a fully separate file/experiment means splits_dir/{experiment_name}
    # resolves the same way build_dataset.py's does, with no special-cased
    # suffix anywhere downstream.
    build_team_stats(config, raw_season_results)

    team_stats, seeds, tourney_results = load_upset_data(config, raw_tourney_results, raw_seeds)
    build_upset_splits(config, team_stats, seeds, tourney_results)