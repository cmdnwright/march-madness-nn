from src.data.dataset import load_data, build_matchups, build_splits
from src.data.preprocess import build_team_stats
from src.data.ingest import load_raw_data
from src.utils.config import load_config

if __name__ == '__main__':
    config = load_config('configs/season.yaml')

    data = load_raw_data(config['data']['raw_dir'])
    raw_season_results = data['season_results'].copy()
    raw_tourney_results = data['tourney_results'].copy()
    raw_seeds = data['seeds'].copy()

    build_team_stats(config, raw_season_results)

    team_stats, seeds, tourney_results = load_data(config, raw_tourney_results, raw_seeds)

    build_splits(config, team_stats, seeds, tourney_results)