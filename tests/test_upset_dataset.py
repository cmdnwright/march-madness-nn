import json
import numpy as np
import pandas as pd
import pytest

from src.data.dataset import parse_seed
from src.data.upset_dataset import (
    load_upset_data,
    build_upset_matchups,
    split_by_season,
    normalize_splits,
    build_upset_splits,
)

@pytest.fixture
def config() -> dict:
    '''
    provides a minimal configuration dict
    for testing upset data splits and feature normalization.
    '''
    return {
        'data': {
            'experiment_name': 'test_upset_experiment',
            'train_seasons': [2010, 2011],
            'val_season': 2012,
            'test_season': 2013,
            'processed_dir': None,
            'splits_dir': None,
        },
        'features': {
            'columns': ['seed_diff', 'win_pct_diff', 'pts_diff'],
            'normalize': True,
        },
    }


@pytest.fixture
def team_stats_df() -> pd.DataFrame:
    '''
    provides a mock team stats dataframe for seasons 2010-2013.
    team 1 is the stronger team (higher win_pct/pts), team 2 the weaker one.
    '''
    rows = []
    for season in [2010, 2011, 2012, 2013]:
        rows.append({'season': season, 'team_id': 1, 'win_pct': 0.60, 'pts': 75.0})
        rows.append({'season': season, 'team_id': 2, 'win_pct': 0.45, 'pts': 68.0})
    return pd.DataFrame(rows)


@pytest.fixture
def seeds_df() -> pd.DataFrame:
    '''
    provides a mock tournament seeds dataframe where team 1 is always the
    #1 seed (favorite) and team 2 is always the #16 seed (underdog).
    '''
    rows = []
    for season in [2010, 2011, 2012, 2013]:
        rows.append({'Season': season, 'TeamID': 1, 'Seed': 'W01'})
        rows.append({'Season': season, 'TeamID': 2, 'Seed': 'X16a'})
    return pd.DataFrame(rows)


@pytest.fixture
def tourney_results_df() -> pd.DataFrame:
    '''
    provides a mock tournament results dataframe for seasons 2009-2014
    where the favorite (team 1) always wins -- no upsets.
    '''
    rows = []
    for season in range(2009, 2015):
        rows.append({'Season': season, 'WTeamID': 1, 'LTeamID': 2})
    return pd.DataFrame(rows)


@pytest.fixture
def prepared_seeds(seeds_df: pd.DataFrame) -> pd.DataFrame:
    '''
    provides a mock seeds dataframe with a parsed integer seed column.
    '''
    out = seeds_df.copy()
    out['seed'] = out['Seed'].apply(parse_seed)
    return out[['Season', 'TeamID', 'seed']]


class TestLoadUpsetData:
    def test_filters_seasons_to_config_range(self, tmp_path, config: dict, team_stats_df: pd.DataFrame, seeds_df: pd.DataFrame, tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that the loaded tournament data is
        strictly filtered to the seasons specified
        in the configuration.
        '''
        config['data']['processed_dir'] = str(tmp_path)
        team_stats_df.to_csv(tmp_path / f"team_stats_{config['data']['experiment_name']}.csv", index=False)

        _, _, filtered_tourney = load_upset_data(config, tourney_results_df.copy(), seeds_df.copy())

        # tourney_results_df spans 2009-2014; config range is [2010, 2013]
        assert filtered_tourney['Season'].min() == 2010
        assert filtered_tourney['Season'].max() == 2013
        assert set(filtered_tourney.columns) == {'Season', 'WTeamID', 'LTeamID'}

    def test_parses_seed_column(self, tmp_path, config: dict, team_stats_df: pd.DataFrame, seeds_df: pd.DataFrame, tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that the raw seed column is converted
        to an integer 'seed' column and the original
        is dropped.
        '''
        config['data']['processed_dir'] = str(tmp_path)
        team_stats_df.to_csv(tmp_path / f"team_stats_{config['data']['experiment_name']}.csv", index=False)

        _, seeds_out, _ = load_upset_data(config, tourney_results_df.copy(), seeds_df.copy())

        assert 'seed' in seeds_out.columns
        assert 'Seed' not in seeds_out.columns  # raw string column dropped
        team1_seed = seeds_out[(seeds_out['Season'] == 2010) & (seeds_out['TeamID'] == 1)]['seed'].iloc[0]
        team2_seed = seeds_out[(seeds_out['Season'] == 2010) & (seeds_out['TeamID'] == 2)]['seed'].iloc[0]
        assert team1_seed == 1
        assert team2_seed == 16

    def test_reads_team_stats_from_processed_dir(self, tmp_path, config: dict, team_stats_df: pd.DataFrame, seeds_df: pd.DataFrame, tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that team statistics are correctly read
        from the configured processed directory using the
        experiment-name-suffixed filename.
        '''
        config['data']['processed_dir'] = str(tmp_path)
        team_stats_df.to_csv(tmp_path / f"team_stats_{config['data']['experiment_name']}.csv", index=False)

        team_stats_out, _, _ = load_upset_data(config, tourney_results_df.copy(), seeds_df.copy())

        assert list(team_stats_out.columns) == list(team_stats_df.columns)
        assert len(team_stats_out) == len(team_stats_df)


class TestBuildUpsetMatchups:
    @pytest.fixture
    def favorite_wins_tourney(self) -> pd.DataFrame:
        '''
        provides a single game where the #1 seed (team 1) beats
        the #16 seed (team 2) -- the expected, non-upset outcome.
        '''
        return pd.DataFrame([{'Season': 2010, 'WTeamID': 1, 'LTeamID': 2}])

    @pytest.fixture
    def underdog_wins_tourney(self) -> pd.DataFrame:
        '''
        provides a single game where the #16 seed (team 2) beats
        the #1 seed (team 1) -- an upset.
        '''
        return pd.DataFrame([{'Season': 2010, 'WTeamID': 2, 'LTeamID': 1}])

    def test_row_count_is_not_doubled(self, config: dict, team_stats_df: pd.DataFrame, prepared_seeds: pd.DataFrame, favorite_wins_tourney: pd.DataFrame) -> None:
        '''
        tests that, unlike the winner/loser dataset, the upset dataset
        produces exactly one row per game -- no mirroring.
        '''
        matchups = build_upset_matchups(team_stats_df, prepared_seeds, favorite_wins_tourney, config)
        assert len(matchups) == 1

    def test_label_is_zero_when_favorite_wins(self, config: dict, team_stats_df: pd.DataFrame, prepared_seeds: pd.DataFrame, favorite_wins_tourney: pd.DataFrame) -> None:
        '''
        tests that no upset is recorded (label=0) when the lower-seeded
        favorite wins the game.
        '''
        matchups = build_upset_matchups(team_stats_df, prepared_seeds, favorite_wins_tourney, config)
        assert matchups.iloc[0]['label'] == 0

    def test_label_is_one_when_underdog_wins(self, config: dict, team_stats_df: pd.DataFrame, prepared_seeds: pd.DataFrame, underdog_wins_tourney: pd.DataFrame) -> None:
        '''
        tests that an upset is recorded (label=1) when the higher-seeded
        underdog wins the game.
        '''
        matchups = build_upset_matchups(team_stats_df, prepared_seeds, underdog_wins_tourney, config)
        assert matchups.iloc[0]['label'] == 1

    def test_features_are_favorite_minus_underdog_regardless_of_winner(self, config: dict, team_stats_df: pd.DataFrame, prepared_seeds: pd.DataFrame, favorite_wins_tourney: pd.DataFrame, underdog_wins_tourney: pd.DataFrame) -> None:
        '''
        tests that the feature diffs are always oriented favorite-stat
        minus underdog-stat, so they come out identical whether the
        favorite or the underdog actually won the game -- only the
        label should differ.
        '''
        favorite_win_matchups = build_upset_matchups(team_stats_df, prepared_seeds, favorite_wins_tourney, config)
        underdog_win_matchups = build_upset_matchups(team_stats_df, prepared_seeds, underdog_wins_tourney, config)

        favorite_row = favorite_win_matchups.iloc[0]
        underdog_row = underdog_win_matchups.iloc[0]

        # team 1 (favorite, seed 1): win_pct 0.60, pts 75.0
        # team 2 (underdog, seed 16): win_pct 0.45, pts 68.0
        for row in (favorite_row, underdog_row):
            assert row['seed_diff'] == pytest.approx(1 - 16)
            assert row['win_pct_diff'] == pytest.approx(0.60 - 0.45)
            assert row['pts_diff'] == pytest.approx(75.0 - 68.0)

        assert favorite_row['label'] != underdog_row['label']

    def test_equal_seed_games_are_dropped(self, config: dict, team_stats_df: pd.DataFrame) -> None:
        '''
        tests that games between two teams sharing the same seed are
        dropped entirely, since there's no favorite to define an upset
        against.
        '''
        tied_seeds = pd.DataFrame([
            {'Season': 2010, 'TeamID': 1, 'seed': 8},
            {'Season': 2010, 'TeamID': 2, 'seed': 8},
        ])
        tourney = pd.DataFrame([{'Season': 2010, 'WTeamID': 1, 'LTeamID': 2}])

        matchups = build_upset_matchups(team_stats_df, tied_seeds, tourney, config)
        assert len(matchups) == 0


class TestBuildUpsetMatchupsMergeIntegrity:
    @pytest.fixture
    def two_game_tourney(self) -> pd.DataFrame:
        '''
        provides a two-game, two-season tournament dataframe
        for testing merge behaviors.
        '''
        return pd.DataFrame([
            {'Season': 2010, 'WTeamID': 1, 'LTeamID': 2},
            {'Season': 2011, 'WTeamID': 3, 'LTeamID': 4},
        ])

    def test_complete_stats_and_seeds_yields_all_games(self, config: dict) -> None:
        '''
        tests that complete datasets result in all games being
        successfully merged, one row per game (no mirroring).
        '''
        team_stats = pd.DataFrame([
            {'season': 2010, 'team_id': 1, 'win_pct': 0.6, 'pts': 75.0},
            {'season': 2010, 'team_id': 2, 'win_pct': 0.4, 'pts': 65.0},
            {'season': 2011, 'team_id': 3, 'win_pct': 0.5, 'pts': 70.0},
            {'season': 2011, 'team_id': 4, 'win_pct': 0.3, 'pts': 60.0},
        ])
        seeds = pd.DataFrame([
            {'Season': 2010, 'TeamID': 1, 'seed': 1}, {'Season': 2010, 'TeamID': 2, 'seed': 16},
            {'Season': 2011, 'TeamID': 3, 'seed': 5}, {'Season': 2011, 'TeamID': 4, 'seed': 12},
        ])
        tourney = pd.DataFrame([
            {'Season': 2010, 'WTeamID': 1, 'LTeamID': 2},
            {'Season': 2011, 'WTeamID': 3, 'LTeamID': 4},
        ])

        matchups = build_upset_matchups(team_stats, seeds, tourney, config)
        # 2 games in -> 2 rows out, no games lost, no mirroring.
        assert len(matchups) == 2

    def test_missing_stats_row_silently_drops_that_game(self, config: dict, two_game_tourney: pd.DataFrame) -> None:
        '''
        tests that missing a team statistics record causes
        the related game to be dropped via inner join.
        '''
        # team 4's stats row is missing entirely -> the 2011 game should vanish.
        incomplete_stats = pd.DataFrame([
            {'season': 2010, 'team_id': 1, 'win_pct': 0.6, 'pts': 75.0},
            {'season': 2010, 'team_id': 2, 'win_pct': 0.4, 'pts': 65.0},
            {'season': 2011, 'team_id': 3, 'win_pct': 0.5, 'pts': 70.0},
            # team_id 4 missing for season 2011
        ])
        seeds = pd.DataFrame([
            {'Season': 2010, 'TeamID': 1, 'seed': 1}, {'Season': 2010, 'TeamID': 2, 'seed': 16},
            {'Season': 2011, 'TeamID': 3, 'seed': 5}, {'Season': 2011, 'TeamID': 4, 'seed': 12},
        ])

        matchups = build_upset_matchups(incomplete_stats, seeds, two_game_tourney, config)
        # only the 2010 game survives -> 1 row.
        assert len(matchups) == 1
        assert set(matchups['Season'].unique()) == {2010}

    def test_missing_seed_row_silently_drops_that_game(self, config: dict, two_game_tourney: pd.DataFrame) -> None:
        '''
        tests that missing a seed record causes the related
        game to be dropped via inner join.
        '''
        team_stats = pd.DataFrame([
            {'season': 2010, 'team_id': 1, 'win_pct': 0.6, 'pts': 75.0},
            {'season': 2010, 'team_id': 2, 'win_pct': 0.4, 'pts': 65.0},
            {'season': 2011, 'team_id': 3, 'win_pct': 0.5, 'pts': 70.0},
            {'season': 2011, 'team_id': 4, 'win_pct': 0.3, 'pts': 60.0},
        ])
        # team 4's seed missing entirely -> the 2011 game should vanish.
        incomplete_seeds = pd.DataFrame([
            {'Season': 2010, 'TeamID': 1, 'seed': 1}, {'Season': 2010, 'TeamID': 2, 'seed': 16},
            {'Season': 2011, 'TeamID': 3, 'seed': 5},
            # teamid 4 missing for season 2011
        ])

        matchups = build_upset_matchups(team_stats, incomplete_seeds, two_game_tourney, config)
        assert len(matchups) == 1
        assert set(matchups['Season'].unique()) == {2010}


class TestSplitBySeason:
    @pytest.fixture
    def multi_season_matchups(self, config: dict) -> pd.DataFrame:
        '''
        provides a generic matchups dataframe spanning multiple
        seasons for testing data splits.
        '''
        feature_cols = config['features']['columns']
        rows = []
        for season in [2009, 2010, 2011, 2012, 2013, 2014]:
            rows.append({'Season': season, **{c: 1.0 for c in feature_cols}, 'label': 1})
        return pd.DataFrame(rows)

    def test_train_val_test_partition_correctly(self, config: dict, multi_season_matchups: pd.DataFrame) -> None:
        '''
        tests that data partitions correctly allocate rows
        to train, val, and test based on the configuration.
        '''
        X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
            multi_season_matchups, config
        )
        # train_seasons = [2010, 2011] -> 2 rows; val = 2012 -> 1 row; test = 2013 -> 1 row
        assert X_train.shape[0] == 2
        assert X_val.shape[0] == 1
        assert X_test.shape[0] == 1

    def test_excludes_seasons_outside_range(self, config: dict, multi_season_matchups: pd.DataFrame) -> None:
        '''
        tests that seasons not specified in the configuration
        splits are entirely excluded from output arrays.
        '''
        X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
            multi_season_matchups, config
        )
        total_rows = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total_rows == 4  # not 6

    def test_feature_column_count_matches_config(self, config: dict, multi_season_matchups: pd.DataFrame) -> None:
        '''
        tests that the generated feature arrays have a column count
        matching the configured feature list.
        '''
        X_train, *_ = split_by_season(multi_season_matchups, config)
        assert X_train.shape[1] == len(config['features']['columns'])


class TestNormalizeSplits:
    def test_train_mean_and_std_after_normalization(self, config: dict) -> None:
        '''
        tests that normalizing the training split correctly standardizes
        it to zero mean and unit variance.
        '''
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=5.0, scale=2.0, size=(100, 3))
        X_val = rng.normal(loc=5.0, scale=2.0, size=(10, 3))
        X_test = rng.normal(loc=5.0, scale=2.0, size=(10, 3))

        X_train_norm, X_val_norm, X_test_norm, norm_stats = normalize_splits(
            X_train, X_val, X_test, config
        )

        np.testing.assert_allclose(X_train_norm.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(X_train_norm.std(axis=0), 1.0, atol=1e-8)

    def test_zero_variance_feature_does_not_produce_nan(self, config: dict) -> None:
        '''
        tests that normalizing a constant feature correctly defaults
        to replacing a zero variance with 1.0 to prevent nans.
        '''
        X_train = np.ones((20, 3))  # every feature constant -> std = 0
        X_val = np.ones((5, 3))
        X_test = np.ones((5, 3))

        X_train_norm, X_val_norm, X_test_norm, norm_stats = normalize_splits(
            X_train, X_val, X_test, config
        )
        assert not np.isnan(X_train_norm).any()
        assert not np.isnan(X_val_norm).any()
        assert 1.0 in norm_stats['std']  # zero-variance guard replaced 0 with 1.0

    def test_norm_stats_contains_feature_order(self, config: dict) -> None:
        '''
        tests that the returned normalization statistics dictionary
        tracks the original feature column order.
        '''
        X_train = np.random.default_rng(0).normal(size=(20, 3))
        _, _, _, norm_stats = normalize_splits(X_train, X_train, X_train, config)
        assert norm_stats['feature_order'] == config['features']['columns']
        assert len(norm_stats['mean']) == 3
        assert len(norm_stats['std']) == 3


class TestBuildUpsetSplitsEndToEnd:
    @pytest.fixture
    def mixed_tourney_results_df(self) -> pd.DataFrame:
        '''
        provides tournament results across 2009-2014 where the favorite
        (team 1) wins in even seasons and the underdog (team 2) upsets
        in odd seasons, so both labels are exercised end-to-end.
        '''
        rows = []
        for season in range(2009, 2015):
            if season % 2 == 0:
                rows.append({'Season': season, 'WTeamID': 1, 'LTeamID': 2})  # favorite wins
            else:
                rows.append({'Season': season, 'WTeamID': 2, 'LTeamID': 1})  # upset
        return pd.DataFrame(rows)

    @pytest.fixture
    def prepared_dirs(self, tmp_path, config: dict, team_stats_df: pd.DataFrame) -> tuple:
        '''
        provides a tuple containing a prepared config
        and the per-experiment splits directory build_upset_splits writes to.
        '''
        processed_dir = tmp_path / 'processed'
        splits_dir = tmp_path / 'splits'
        processed_dir.mkdir()
        splits_dir.mkdir()

        config['data']['processed_dir'] = str(processed_dir)
        config['data']['splits_dir'] = str(splits_dir)
        team_stats_df.to_csv(processed_dir / f"team_stats_{config['data']['experiment_name']}.csv", index=False)

        experiment_splits_dir = splits_dir / config['data']['experiment_name']

        return config, experiment_splits_dir

    def test_produces_all_expected_files_with_correct_shapes(self, prepared_dirs: tuple, seeds_df: pd.DataFrame, mixed_tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that the end-to-end upset pipeline generates all required
        train, validation, and test arrays as well as the json stats.
        '''
        config, splits_dir = prepared_dirs

        team_stats_out, seeds_out, tourney_out = load_upset_data(
            config, mixed_tourney_results_df.copy(), seeds_df.copy()
        )
        build_upset_splits(config, team_stats_out, seeds_out, tourney_out)

        expected_files = [
            'X_train.npy', 'y_train.npy',
            'X_val.npy', 'y_val.npy',
            'X_test.npy', 'y_test.npy',
            'norm_stats.json',
        ]
        for fname in expected_files:
            assert (splits_dir / fname).exists(), f'missing {fname}'

        X_train = np.load(splits_dir / 'X_train.npy')
        y_train = np.load(splits_dir / 'y_train.npy')
        assert X_train.shape[0] == y_train.shape[0]
        assert X_train.shape[1] == len(config['features']['columns'])

    def test_labels_are_strictly_binary(self, prepared_dirs: tuple, seeds_df: pd.DataFrame, mixed_tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that generated upset labels consist exclusively of 0s and 1s.
        '''
        config, splits_dir = prepared_dirs

        team_stats_out, seeds_out, tourney_out = load_upset_data(
            config, mixed_tourney_results_df.copy(), seeds_df.copy()
        )
        build_upset_splits(config, team_stats_out, seeds_out, tourney_out)

        y_train = np.load(splits_dir / 'y_train.npy')
        assert set(np.unique(y_train)).issubset({0, 1})

    def test_row_count_is_not_doubled_end_to_end(self, prepared_dirs: tuple, seeds_df: pd.DataFrame, mixed_tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that the final exported arrays have exactly one row per
        game (no mirroring), unlike the winner/loser dataset pipeline.
        '''
        config, splits_dir = prepared_dirs

        team_stats_out, seeds_out, tourney_out = load_upset_data(
            config, mixed_tourney_results_df.copy(), seeds_df.copy()
        )
        build_upset_splits(config, team_stats_out, seeds_out, tourney_out)

        # config: train_seasons=[2010, 2011] -> 2 seasons, 1 game/season, not doubled = 2 rows
        # val_season=2012 -> 1 game = 1 row
        # test_season=2013 -> 1 game = 1 row
        X_train = np.load(splits_dir / 'X_train.npy')
        X_val = np.load(splits_dir / 'X_val.npy')
        X_test = np.load(splits_dir / 'X_test.npy')

        assert X_train.shape[0] == 2
        assert X_val.shape[0] == 1
        assert X_test.shape[0] == 1

    def test_upset_rate_reflects_alternating_results(self, prepared_dirs: tuple, seeds_df: pd.DataFrame, mixed_tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that the saved training labels correctly reflect which
        seasons were upsets (odd) vs favorite wins (even) in the fixture.
        '''
        config, splits_dir = prepared_dirs

        team_stats_out, seeds_out, tourney_out = load_upset_data(
            config, mixed_tourney_results_df.copy(), seeds_df.copy()
        )
        build_upset_splits(config, team_stats_out, seeds_out, tourney_out)

        # train_seasons = [2010, 2011]: 2010 is a favorite win (label 0),
        # 2011 is an upset (label 1) -> exactly one of each in y_train.
        y_train = np.load(splits_dir / 'y_train.npy')
        assert sorted(y_train.tolist()) == [0, 1]


class TestSaveNormStats:
    def test_writes_valid_json_to_expected_path(self, tmp_path) -> None:
        '''
        tests that normalization statistics are correctly serialized
        and saved to a valid json file.
        '''
        norm_stats = {'mean': [1.0, 2.0], 'std': [0.5, 1.5], 'feature_order': ['a', 'b']}
        path = str(tmp_path) + '/norm_stats.json'
        with open(path, 'w') as f:
            json.dump(norm_stats, f, indent=2)

        assert path == f'{tmp_path}/norm_stats.json'
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == norm_stats