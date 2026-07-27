import pandas as pd
import json
import numpy as np
from src.data.ingest import load_raw_data
import re

def parse_seed(seed_str: str) -> int:
    match = re.search(r'\d+', seed_str)
    if not match:
        raise ValueError('no seed found')
    return int(match.group())

def load_data(config, tourney_results, seeds): 
    team_stats = pd.read_csv(f'{config["data"]["processed_dir"]}/team_stats.csv')

    seeds['seed'] = seeds['Seed'].apply(parse_seed)
    seeds = seeds[['Season', 'TeamID', 'seed']]

    min_season = config['data']['train_seasons'][0]
    max_season = config['data']['test_season']
    tourney_results = tourney_results[
        (tourney_results['Season'] >= min_season) &
        (tourney_results['Season'] <= max_season)
    ]

    return team_stats, seeds, tourney_results[['Season', 'WTeamID', 'LTeamID']]

def build_matchups(team_stats_df, seed_df, tournament_df, config):
    winner_matchups = pd.merge(
        left=tournament_df,
        right=team_stats_df,
        left_on=['Season', 'WTeamID'],
        right_on=['season', 'team_id'],
    )

    all_matchups = pd.merge(
        left=winner_matchups,
        right=team_stats_df,
        left_on=['Season', 'LTeamID'],
        right_on=['season', 'team_id'],
        suffixes=(None, '_l')
    )

    all_matchups = pd.merge(
        left=all_matchups,
        right=seed_df,
        left_on= ['Season', 'WTeamID'],
        right_on= ['Season', 'TeamID']
    )

    all_matchups = pd.merge(
            left=all_matchups,
            right=seed_df,
            left_on= ['Season', 'LTeamID'],
            right_on= ['Season', 'TeamID'],
            suffixes=(None, '_l')
        )

    feature_cols = config['features']['columns']

    for feature in feature_cols:
        df_column = feature.replace('_diff', '')
        all_matchups[feature] = all_matchups[df_column] - all_matchups[f'{df_column}_l']
    
    cols = ['Season'] + feature_cols
    matchups = all_matchups[cols].copy()
    matchups['label'] = 1

    mirrored = matchups.copy()
    mirrored[feature_cols] = -mirrored[feature_cols]
    mirrored['label'] = 1 - mirrored['label']

    matchups = pd.concat([matchups, mirrored], ignore_index=True)

    return matchups

def split_by_season(matchups, config):
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

def normalize_splits(X_train, X_val, X_test, config):
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

def save_norm_stats(norm_stats, splits_dir):
    path = f'{splits_dir}/norm_stats.json'
    with open(path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    return path

def build_splits(config, team_stats, seeds, tourney_results):

    matchups = build_matchups(team_stats, seeds, tourney_results, config)

    X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(matchups, config)
    X_train, X_val, X_test, norm_stats = normalize_splits(X_train, X_val, X_test, config)

    splits_dir = config['data']['splits_dir']
    np.save(f'{splits_dir}/X_train.npy', X_train)
    np.save(f'{splits_dir}/y_train.npy', y_train)
    np.save(f'{splits_dir}/X_val.npy', X_val)
    np.save(f'{splits_dir}/y_val.npy', y_val)
    np.save(f'{splits_dir}/X_test.npy', X_test)
    np.save(f'{splits_dir}/y_test.npy', y_test)

    save_norm_stats(norm_stats, splits_dir)

    print(f'Saved splits to {splits_dir}: '
          f'train={X_train.shape}, val={X_val.shape}, test={X_test.shape}')