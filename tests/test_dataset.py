import json
import numpy as np
import pandas as pd
import pytest

from src.data.dataset import (
    parse_seed,
    load_data,
    build_matchups,
    split_by_season,
    normalize_splits,
    save_norm_stats,
    build_splits,
)

@pytest.fixture
def config() -> dict:
    '''
    provides a minimal configuration dict 
    for testing data splits and feature normalization.
    '''
    return {
        'data': {
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
    provides a mock team stats dataframe 
    for seasons 2010-2013 matching expected 
    preprocessed output.
    '''
    rows = []
    for season in [2010, 2011, 2012, 2013]:
        rows.append({'season': season, 'team_id': 1, 'win_pct': 0.60, 'pts': 75.0})
        rows.append({'season': season, 'team_id': 2, 'win_pct': 0.45, 'pts': 68.0})
    return pd.DataFrame(rows)


@pytest.fixture
def seeds_df() -> pd.DataFrame:
    '''
    provides a mock tournament seeds dataframe 
    containing play-in suffixes in kaggle format.
    '''
    rows = []
    for season in [2010, 2011, 2012, 2013]:
        rows.append({'Season': season, 'TeamID': 1, 'Seed': 'W01'})
        rows.append({'Season': season, 'TeamID': 2, 'Seed': 'X16a'})
    return pd.DataFrame(rows)


@pytest.fixture
def tourney_results_df() -> pd.DataFrame:
    '''
    provides a mock tournament results dataframe 
    for seasons 2009-2014 where team 1 beats team 2.
    '''
    rows = []
    for season in range(2009, 2015):
        rows.append({'Season': season, 'WTeamID': 1, 'LTeamID': 2})
    return pd.DataFrame(rows)

class TestParseSeed:
    def test_simple_seed(self) -> None:
        '''
        tests that a standard region-seed string 
        is parsed to an integer.
        '''
        assert parse_seed('W01') == 1

    def test_double_digit_seed(self) -> None:
        '''
        tests that a two-digit region-seed string 
        is correctly parsed to an integer.
        '''
        assert parse_seed('X16') == 16

    def test_playin_suffix_ignored(self) -> None:
        '''
        tests that character suffixes denoting play-in games 
        are ignored during seed parsing.
        '''
        assert parse_seed('X16a') == 16
        assert parse_seed('Y11b') == 11

    def test_no_digits_raises(self) -> None:
        '''
        tests that parsing a string with no digits 
        raises a valueerror.
        '''
        with pytest.raises(ValueError):
            parse_seed('ABC')

class TestLoadData:
    def test_filters_seasons_to_config_range(self, tmp_path, config: dict, team_stats_df: pd.DataFrame, seeds_df: pd.DataFrame, tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that the loaded tournament data is 
        strictly filtered to the seasons specified 
        in the configuration.
        '''
        config['data']['processed_dir'] = str(tmp_path)
        team_stats_df.to_csv(tmp_path / 'team_stats.csv', index=False)

        _, _, filtered_tourney = load_data(config, tourney_results_df.copy(), seeds_df.copy())

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
        team_stats_df.to_csv(tmp_path / 'team_stats.csv', index=False)

        _, seeds_out, _ = load_data(config, tourney_results_df.copy(), seeds_df.copy())

        assert 'seed' in seeds_out.columns
        assert 'Seed' not in seeds_out.columns  # raw string column dropped
        team1_seed = seeds_out[(seeds_out['Season'] == 2010) & (seeds_out['TeamID'] == 1)]['seed'].iloc[0]
        team2_seed = seeds_out[(seeds_out['Season'] == 2010) & (seeds_out['TeamID'] == 2)]['seed'].iloc[0]
        assert team1_seed == 1
        assert team2_seed == 16

    def test_reads_team_stats_from_processed_dir(self, tmp_path, config: dict, team_stats_df: pd.DataFrame, seeds_df: pd.DataFrame, tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that team statistics are correctly read 
        from the configured processed directory.
        '''
        config['data']['processed_dir'] = str(tmp_path)
        team_stats_df.to_csv(tmp_path / 'team_stats.csv', index=False)

        team_stats_out, _, _ = load_data(config, tourney_results_df.copy(), seeds_df.copy())

        assert list(team_stats_out.columns) == list(team_stats_df.columns)
        assert len(team_stats_out) == len(team_stats_df)

class TestBuildMatchups:
    @pytest.fixture
    def prepared_seeds(self, seeds_df: pd.DataFrame) -> pd.DataFrame:
        '''
        provides a mock seeds dataframe with 
        a parsed integer seed column.
        '''
        out = seeds_df.copy()
        out['seed'] = out['Seed'].apply(parse_seed)
        return out[['Season', 'TeamID', 'seed']]

    @pytest.fixture
    def small_tourney(self) -> pd.DataFrame:
        '''
        provides a single-game tournament dataframe 
        for simplified diff and mirror validation.
        '''
        # one game, one season, to make diff/mirror math easy to verify by hand.
        return pd.DataFrame([{'Season': 2010, 'WTeamID': 1, 'LTeamID': 2}])

    def test_row_count_is_doubled(self, config: dict, team_stats_df: pd.DataFrame, prepared_seeds: pd.DataFrame, small_tourney: pd.DataFrame) -> None:
        '''
        tests that building matchups creates two rows 
        (original and mirrored) for every one game.
        '''
        matchups = build_matchups(team_stats_df, prepared_seeds, small_tourney, config)
        # one game in -> one original row + one mirrored row
        assert len(matchups) == 2

    def test_label_balance_is_exact(self, config: dict, team_stats_df: pd.DataFrame, prepared_seeds: pd.DataFrame, small_tourney: pd.DataFrame) -> None:
        '''
        tests that the mirrored data generation maintains 
        a perfectly balanced label distribution.
        '''
        matchups = build_matchups(team_stats_df, prepared_seeds, small_tourney, config)
        counts = matchups['label'].value_counts()
        assert counts.get(1) == 1
        assert counts.get(0) == 1

    def test_diff_values_correct(self, config: dict, team_stats_df: pd.DataFrame, prepared_seeds: pd.DataFrame, small_tourney: pd.DataFrame) -> None:
        '''
        tests that the calculated differences for features 
        (seed, win percentage, points) match expected arithmetic.
        '''
        matchups = build_matchups(team_stats_df, prepared_seeds, small_tourney, config)
        original = matchups[matchups['label'] == 1].iloc[0]

        # team 1 (winner): seed 1, win_pct 0.60, pts 75.0
        # team 2 (loser):  seed 16, win_pct 0.45, pts 68.0
        assert original['seed_diff'] == pytest.approx(1 - 16)
        assert original['win_pct_diff'] == pytest.approx(0.60 - 0.45)
        assert original['pts_diff'] == pytest.approx(75.0 - 68.0)

    def test_mirrored_row_is_negation_with_flipped_label(self, config: dict, team_stats_df: pd.DataFrame, prepared_seeds: pd.DataFrame, small_tourney: pd.DataFrame) -> None:
        '''
        tests that the mirrored matchup row correctly negates 
        feature values and flips the target label.
        '''
        matchups = build_matchups(team_stats_df, prepared_seeds, small_tourney, config)
        original = matchups[matchups['label'] == 1].iloc[0]
        mirrored = matchups[matchups['label'] == 0].iloc[0]

        for col in config['features']['columns']:
            assert mirrored[col] == pytest.approx(-original[col])
        assert mirrored['Season'] == original['Season']

class TestBuildMatchupsMergeIntegrity:
    @pytest.fixture
    def two_game_tourney(self) -> pd.DataFrame:
        '''
        provides a two-game, two-season tournament dataframe 
        for testing merge behaviors.
        '''
        # two independent games, two different seasons.
        return pd.DataFrame([
            {'Season': 2010, 'WTeamID': 1, 'LTeamID': 2},
            {'Season': 2011, 'WTeamID': 3, 'LTeamID': 4},
        ])

    def test_complete_stats_and_seeds_yields_all_games_doubled(self, config: dict) -> None:
        '''
        tests that complete datasets result in all games 
        being successfully merged and doubled.
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

        matchups = build_matchups(team_stats, seeds, tourney, config)
        # 2 games in -> 2 original + 2 mirrored = 4 rows out. no games lost.
        assert len(matchups) == 4

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

        matchups = build_matchups(incomplete_stats, seeds, two_game_tourney, config)
        # only the 2010 game survives -> 1 original + 1 mirrored = 2 rows.
        # known gap: this drop happens silently (inner join), with no error raised.
        assert len(matchups) == 2
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

        matchups = build_matchups(team_stats, incomplete_seeds, two_game_tourney, config)
        assert len(matchups) == 2
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
        # 2009 and 2014 belong to none of train/val/test
        X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
            multi_season_matchups, config
        )
        total_rows = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total_rows == 4 # not 6

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

    def test_val_and_test_use_train_stats_not_their_own(self, config: dict) -> None:
        '''
        tests that validation and test sets are normalized using the 
        training set statistics rather than their own.
        '''
        # train has a different mean than val -> val_norm should not be centered at 0
        X_train = np.full((50, 3), 5.0)
        X_val = np.full((10, 3), 50.0)
        X_test = np.full((10, 3), 5.0)

        # inject a bit of spread so std isn't zero
        X_train[0] += 1.0

        X_train_norm, X_val_norm, X_test_norm, norm_stats = normalize_splits(
            X_train, X_val, X_test, config
        )
        # val was 10x train's value -> after applying train's mean/std it should be far from 0
        assert np.all(X_val_norm > 5)

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

class TestSaveNormStats:
    def test_writes_valid_json_to_expected_path(self, tmp_path) -> None:
        '''
        tests that normalization statistics are correctly serialized 
        and saved to a valid json file.
        '''
        norm_stats = {'mean': [1.0, 2.0], 'std': [0.5, 1.5], 'feature_order': ['a', 'b']}
        path = save_norm_stats(norm_stats, str(tmp_path))

        assert path == f'{tmp_path}/norm_stats.json'
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == norm_stats

class TestBuildSplitsEndToEnd:
    @pytest.fixture
    def prepared_dirs(self, tmp_path, config: dict, team_stats_df: pd.DataFrame) -> tuple:
        '''
        provides a tuple containing a prepared config 
        and temporary splits directory for end-to-end tests.
        '''
        processed_dir = tmp_path / 'processed'
        splits_dir = tmp_path / 'splits'
        processed_dir.mkdir()
        splits_dir.mkdir()

        config['data']['processed_dir'] = str(processed_dir)
        config['data']['splits_dir'] = str(splits_dir)
        team_stats_df.to_csv(processed_dir / 'team_stats.csv', index=False)

        return config, splits_dir

    def test_produces_all_expected_files_with_correct_shapes(self, prepared_dirs: tuple, seeds_df: pd.DataFrame, tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that the end-to-end pipeline generates all required train, 
        validation, and test arrays as well as the json stats.
        '''
        config, splits_dir = prepared_dirs

        team_stats_out, seeds_out, tourney_out = load_data(
            config, tourney_results_df.copy(), seeds_df.copy()
        )
        build_splits(config, team_stats_out, seeds_out, tourney_out)

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

    def test_labels_are_strictly_binary(self, prepared_dirs: tuple, seeds_df: pd.DataFrame, tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that generated model target labels consist 
        exclusively of 0s and 1s.
        '''
        config, splits_dir = prepared_dirs

        team_stats_out, seeds_out, tourney_out = load_data(
            config, tourney_results_df.copy(), seeds_df.copy()
        )
        build_splits(config, team_stats_out, seeds_out, tourney_out)

        y_train = np.load(splits_dir / 'y_train.npy')
        assert set(np.unique(y_train)).issubset({0, 1})

    def test_train_is_normalized_to_zero_mean(self, prepared_dirs: tuple, seeds_df: pd.DataFrame, tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that the final saved training feature matrix has 
        a mean of zero across all columns.
        '''
        config, splits_dir = prepared_dirs

        team_stats_out, seeds_out, tourney_out = load_data(
            config, tourney_results_df.copy(), seeds_df.copy()
        )
        build_splits(config, team_stats_out, seeds_out, tourney_out)

        X_train = np.load(splits_dir / 'X_train.npy')
        np.testing.assert_allclose(X_train.mean(axis=0), 0.0, atol=1e-6)

    def test_row_count_matches_expected_given_two_teams_across_seasons(self, prepared_dirs: tuple, seeds_df: pd.DataFrame, tourney_results_df: pd.DataFrame) -> None:
        '''
        tests that the final exported arrays have the correct exact 
        row counts based on the config seasons and doubling logic.
        '''
        config, splits_dir = prepared_dirs

        team_stats_out, seeds_out, tourney_out = load_data(
            config, tourney_results_df.copy(), seeds_df.copy()
        )
        build_splits(config, team_stats_out, seeds_out, tourney_out)

        # config: train_seasons=[2010, 2011] -> 2 seasons, 1 game/season, doubled = 4 rows
        # val_season=2012 -> 1 game, doubled = 2 rows
        # test_season=2013 -> 1 game, doubled = 2 rows
        X_train = np.load(splits_dir / 'X_train.npy')
        X_val = np.load(splits_dir / 'X_val.npy')
        X_test = np.load(splits_dir / 'X_test.npy')

        assert X_train.shape[0] == 4
        assert X_val.shape[0] == 2
        assert X_test.shape[0] == 2