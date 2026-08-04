import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.dataset import parse_seed


def load_upset_data(config: dict, tourney_results: pd.DataFrame, seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''converts the raw tournament results, regular season team stats, and seeds into usable forms.

    same loading logic as dataset.load_data, kept separate so the upset pipeline can evolve
    independently (e.g. different season windows)

    Parameters
    ----------
    config : dict
        config of the experiement, to access the team stats csv
    tourney_results : pd.DataFrame
        raw tournament resutls df from kaggle
    seeds : pd.DataFrame
        raw seeds df from kaggle

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        team stats df (aggregated stats per team per season), seeds df (team id to integer seed per season),
        tourney results df (tournament games team ids winner and loser)
    '''
    experiment_name = config['data']['experiment_name']
    team_stats = pd.read_csv(f'{config["data"]["processed_dir"]}/team_stats_{experiment_name}.csv')

    seeds = seeds.copy()
    seeds['seed'] = seeds['Seed'].apply(parse_seed)
    seeds = seeds[['Season', 'TeamID', 'seed']]

    min_season = config['data']['train_seasons'][0]
    max_season = config['data']['test_season']
    tourney_results = tourney_results[
        (tourney_results['Season'] >= min_season) &
        (tourney_results['Season'] <= max_season)
    ]

    return team_stats, seeds, tourney_results[['Season', 'WTeamID', 'LTeamID']]


def build_upset_matchups(team_stats_df: pd.DataFrame, seed_df: pd.DataFrame, tournament_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    '''builds match up data for all tournament games in all seasons, oriented favorite-vs-underdog.

    one row per game, oriented favorite-vs-underdog (by seed) rather than winner-vs-loser.
    label=1 means the underdog won (an upset happened)

    games where both teams share a seed are dropped: there's no favorite to define an upset against

    Parameters
    ----------
    team_stats_df : pd.DataFrame
        aggregated season stats per team per season
    seed_df : pd.DataFrame
        seeds to team ids including integer seeds
    tournament_df : pd.DataFrame
        tournament results df formatted by season and WTeamID LTeamID
    config : dict
        config for the data experiment, to build the correct features

    Returns
    -------
    pd.DataFrame
        data for models of the form season, features, label
    '''
    # merge team stats into tournament results on winner and loser ids to get agg stats from reg season per team per game
    winner_matchups = pd.merge(
        left=tournament_df, right=team_stats_df,
        left_on=['Season', 'WTeamID'], right_on=['season', 'team_id'],
    )
    all_matchups = pd.merge(
        left=winner_matchups, right=team_stats_df,
        left_on=['Season', 'LTeamID'], right_on=['season', 'team_id'],
        suffixes=(None, '_l')
    )
    all_matchups = pd.merge(
        left=all_matchups, right=seed_df,
        left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID']
    )
    all_matchups = pd.merge(
        left=all_matchups, right=seed_df,
        left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'],
        suffixes=(None, '_l')
    )
    # Drop games with no seed favorite (identical seed on both sides).
    all_matchups = all_matchups[all_matchups['seed'] != all_matchups['seed_l']].copy()

    winner_is_favorite = all_matchups['seed'] < all_matchups['seed_l']

    feature_cols = config['features']['columns']
    diff_features = [f for f in feature_cols if f != 'season_norm']

    matchups = pd.DataFrame({'Season': all_matchups['Season']})

    # create difference features, oriented favorite_stat - underdog_stat regardless of who won
    for feature in diff_features:
        df_column = feature.replace('_diff', '')
        winner_stat = all_matchups[df_column]
        loser_stat = all_matchups[f'{df_column}_l']
        # favorite_stat - underdog_stat, regardless of who actually won
        matchups[feature] = np.where(
            winner_is_favorite, winner_stat - loser_stat, loser_stat - winner_stat
        )

    # norm to prevent magnitude imbalance baising feature importance
    if 'season_norm' in feature_cols:
        min_season = config['data']['train_seasons'][0]
        max_season = config['data']['test_season']
        matchups['season_norm'] = (
            (all_matchups['Season'] - min_season) / (max_season - min_season)
        )

    # underdog won <=> the favorite was NOT the winner
    matchups['label'] = (~winner_is_favorite).astype(int).to_numpy()

    return matchups[['Season'] + feature_cols + ['label']]


def split_by_season(matchups: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''takes the match up data with seasons features and labels and splits them by season into train test val splits

    Parameters
    ----------
    matchups : pd.DataFrame
        dataframe of season, features, label
    config : dict
        config of the experiment for feature list, target seasons

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        training, val, and test x features, label; features shape (N, F), label shape (N,)
    '''
    feature_cols = config['features']['columns']
    data_config = config['data']
    train_start, train_end = data_config['train_seasons']
    val_season = data_config['val_season']
    test_season = data_config['test_season']

    train_df = matchups[(matchups['Season'] >= train_start) & (matchups['Season'] <= train_end)]
    val_df = matchups[matchups['Season'] == val_season]
    test_df = matchups[matchups['Season'] == test_season]

    X_train, y_train = train_df[feature_cols].to_numpy(), train_df['label'].to_numpy()
    X_val, y_val = val_df[feature_cols].to_numpy(), val_df['label'].to_numpy()
    X_test, y_test = test_df[feature_cols].to_numpy(), test_df['label'].to_numpy()

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_splits(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list]]:
    '''normalizes the feature sets using only the statistics of the train set to prevent data leakage

    Parameters
    ----------
    X_train : np.ndarray
        train features (N, F)
    X_val : np.ndarray
        val features (N, F)
    X_test : np.ndarray
        test features (N, F)
    config : dict
        config of the experiemnt for feature order

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list]]
        normalized splits and the summary statistics of the normalization
    '''
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    norm_stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'feature_order': config['features']['columns']
    }
    return X_train_norm, X_val_norm, X_test_norm, norm_stats


def build_upset_splits(config: dict, team_stats: pd.DataFrame, seeds: pd.DataFrame, tourney_results: pd.DataFrame) -> None:
    '''builds the full upset splits from curated data

    Parameters
    ----------
    config : dict
        config of the data experiment for save location
    team_stats_df : pd.DataFrame
            aggregated season stats per team per season
    seed_df : pd.DataFrame
        seeds to team ids including integer seeds
    tournament_df : pd.DataFrame
        tournament results df formatted by season and WTeamID LTeamID
    '''
    matchups = build_upset_matchups(team_stats, seeds, tourney_results, config)

    X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(matchups, config)
    X_train, X_val, X_test, norm_stats = normalize_splits(X_train, X_val, X_test, config)

    experiment_name = config['data']['experiment_name']
    splits_dir = Path(f'{config["data"]["splits_dir"]}/{experiment_name}')
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.save(f'{splits_dir}/X_train.npy', X_train)
    np.save(f'{splits_dir}/y_train.npy', y_train)
    np.save(f'{splits_dir}/X_val.npy', X_val)
    np.save(f'{splits_dir}/y_val.npy', y_val)
    np.save(f'{splits_dir}/X_test.npy', X_test)
    np.save(f'{splits_dir}/y_test.npy', y_test)

    path = splits_dir / 'norm_stats.json'
    with open(path, 'w') as f:
        json.dump(norm_stats, f, indent=2)

    upset_rate_train = y_train.mean() if len(y_train) else float('nan')
    print(f'Saved upset splits to {splits_dir}: '
          f'train={X_train.shape} (upset rate {upset_rate_train:.3f}), '
          f'val={X_val.shape}, test={X_test.shape}')