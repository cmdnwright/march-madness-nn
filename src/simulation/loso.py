import json
import copy
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from src.data.dataset import parse_seed, build_matchups, normalize_splits
from src.data.upset_dataset import build_upset_matchups
from src.model.model import MatchupClassifier
from src.training.train import train_model
from src.simulation.bracket import (
    build_bracket_tree,
    chalk_predict_fn,
    random_predict_fn,
    classical_predict_fn,
    nn_predict_fn,
    hybrid_predict_fn,
    simulate_bracket,
    resolve_actual_winners,
    score_simulations,
    ROUND_WEIGHTS,
)

DEFAULT_MODEL_BUILDERS: dict[str, Callable[[], Any]] = {
    'lr': lambda: LogisticRegression(),
    'rf': lambda: RandomForestClassifier(random_state=0),
}


def check_loso_compatible(config: dict) -> None:
    '''raises if a config uses a feature that isn't well defined under leave-one-season-out

    Parameters
    ----------
    config : dict
        config of the experiment to check

    Raises
    ------
    ValueError
        config['features']['columns'] includes 'season_norm', which has no
        well-defined meaning under LOSO
    '''
    if 'season_norm' in config['features']['columns']:
        raise ValueError(
            "season_norm is not well-defined under LOSO (no single contiguous "
            "train range once a season is pulled from the middle of the pool) "
            "-- remove it from this config's features.columns before running "
            "the LOSO backtest, or add an explicit LOSO-safe definition here."
        )


def validate_shared_feature_set(configs: list[dict]) -> None:
    '''raises if the configs passed to run_loso_backtest don't share a feature set or data source

    LR/RF and the matchup pool are shared across configs, so every config must
    agree on features.columns and data.experiment_name, and every model must
    have a distinct model.experiment_name to label its own results row

    Parameters
    ----------
    configs : list[dict]
        configs to validate, one per NN variant being compared

    Raises
    ------
    ValueError
        configs disagree on features.columns or data.experiment_name, or two
        configs share the same model.experiment_name
    '''
    base = configs[0]
    base_cols = sorted(base['features']['columns'])
    base_data_name = base['data']['experiment_name']
    for cfg in configs[1:]:
        if sorted(cfg['features']['columns']) != base_cols:
            raise ValueError(
                "All configs passed to run_loso_backtest must share the same "
                "features.columns -- got a mismatch between "
                f"{base['model']['experiment_name']!r} and "
                f"{cfg['model']['experiment_name']!r}. LR/RF and the matchup "
                "pool are shared across configs, so a different feature set "
                "isn't supported here."
            )
        if cfg['data']['experiment_name'] != base_data_name:
            raise ValueError(
                "All configs passed to run_loso_backtest must share the same "
                "data.experiment_name (they read the same team_stats file) -- "
                f"got {base_data_name!r} vs {cfg['data']['experiment_name']!r}."
            )

    model_names = [cfg['model']['experiment_name'] for cfg in configs]
    if len(set(model_names)) != len(model_names):
        raise ValueError(
            f"configs have duplicate model.experiment_name values {model_names!r} "
            "-- each NN variant needs a distinct one, since that's what labels "
            "its results row and its checkpoint path."
        )


def prepare_seed_columns(seeds_raw: pd.DataFrame) -> pd.DataFrame:
    '''parses raw kaggle seed codes into integer seeds, keeping only the columns matchup building needs

    Parameters
    ----------
    seeds_raw : pd.DataFrame
        raw seeds df from kaggle

    Returns
    -------
    pd.DataFrame
        seeds df of the form Season, TeamID, seed (int)
    '''
    seeds_for_matchups = seeds_raw.copy()
    seeds_for_matchups['seed'] = seeds_for_matchups['Seed'].apply(parse_seed)
    return seeds_for_matchups[['Season', 'TeamID', 'seed']]


def eligible_loso_seasons(config: dict) -> list[int]:
    '''lists every season in the training window, since LOSO holds out one training season at a time

    Parameters
    ----------
    config : dict
        config of the experiment, to get the training season range

    Returns
    -------
    list[int]
        every season from train_seasons[0] to train_seasons[1] inclusive
    '''
    train_start, train_end = config['data']['train_seasons']
    return list(range(train_start, train_end + 1))


def season_has_bracket_data(season: int, slots_df: pd.DataFrame) -> bool:
    '''checks whether a season has a recorded bracket structure to simulate against

    Parameters
    ----------
    season : int
        season to check
    slots_df : pd.DataFrame
        raw slots df as returned by bracket.load_slots

    Returns
    -------
    bool
        True if slots_df has at least one row for this season
    '''
    return not slots_df[slots_df['Season'] == season].empty


RESULTS_COLUMNS = ['season', 'model', 'mean_score', 'std_score', 'scores']


def load_loso_results(output_path: str | Path) -> pd.DataFrame:
    '''loads previously saved LOSO results, or an empty results frame if none exist yet

    Parameters
    ----------
    output_path : str or Path
        csv path the results are (or will be) saved to

    Returns
    -------
    pd.DataFrame
        results df with columns RESULTS_COLUMNS; 'scores' is parsed back into a list per row
    '''
    output_path = Path(output_path)
    if not output_path.exists():
        return pd.DataFrame(columns=RESULTS_COLUMNS)
    df = pd.read_csv(output_path)
    df['scores'] = df['scores'].apply(json.loads)
    return df


def _append_fold_rows(output_path: str | Path, fold_rows: list[dict]) -> None:
    '''appends one fold's result rows to the results csv, writing a header only if the file is new

    Parameters
    ----------
    output_path : str or Path
        csv path to append to
    fold_rows : list[dict]
        one dict per model, matching RESULTS_COLUMNS, for a single held-out season
    '''
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # scores is a list, which csv can't round-trip on its own so serialize it to a json string per row
    serializable_rows = [dict(row, scores=json.dumps(row['scores'])) for row in fold_rows]
    df = pd.DataFrame(serializable_rows, columns=RESULTS_COLUMNS)
    write_header = not output_path.exists()
    df.to_csv(output_path, mode='a', header=write_header, index=False)


def existing_models_by_season(output_path: str | Path) -> dict[int, set[str]]:
    '''builds a season -> set of already-scored model names lookup, to support resuming a backtest

    Parameters
    ----------
    output_path : str or Path
        csv path previous results were saved to

    Returns
    -------
    dict[int, set[str]]
        season -> set of model names already present in the results csv for that season
    '''
    existing = load_loso_results(output_path)
    if existing.empty:
        return {}
    return {int(season): set(models) for season, models in existing.groupby('season')['model'].apply(set).items()} # type: ignore


def fold_train_val_seasons(all_seasons: list[int], held_out_season: int) -> tuple[list[int], int]:
    '''splits the remaining seasons (after removing the held-out one) into a fold's train seasons and one val season

    the val season is the most recent season strictly before the held-out season, so
    validation never leaks information from the future relative to the held-out season.
    if the held-out season is the earliest in the pool (no earlier seasons exist), the
    most recent season in the remaining pool is used instead

    Parameters
    ----------
    all_seasons : list[int]
        every season eligible for LOSO (see eligible_loso_seasons)
    held_out_season : int
        season being held out as this fold's test season

    Returns
    -------
    tuple[list[int], int]
        remaining seasons to train this fold on, and the season to validate on
    '''
    pool = [s for s in all_seasons if s != held_out_season]
    earlier = [s for s in pool if s < held_out_season]
    val_season = earlier[-1] if earlier else max(pool)
    fold_train_seasons = [s for s in pool if s != val_season]
    return fold_train_seasons, val_season


def build_loso_matchup_pool(config: dict, team_stats: pd.DataFrame, seeds_for_matchups: pd.DataFrame, tourney_results: pd.DataFrame) -> pd.DataFrame:
    '''builds the full winner/loser matchup pool over the training window, shared across every LOSO fold

    building this once up front (rather than per fold) avoids redoing the same
    merges and diff features for every held-out season

    Parameters
    ----------
    config : dict
        config of the experiment, to get the training season range and feature list
    team_stats : pd.DataFrame
        aggregated season stats per team per season
    seeds_for_matchups : pd.DataFrame
        seeds df with parsed integer seeds (see prepare_seed_columns)
    tourney_results : pd.DataFrame
        raw tournament results df from kaggle

    Returns
    -------
    pd.DataFrame
        matchup pool of the form season, features, label, covering every training season
    '''
    train_start, train_end = config['data']['train_seasons']
    pool_results = tourney_results[
        (tourney_results['Season'] >= train_start) & (tourney_results['Season'] <= train_end)
    ][['Season', 'WTeamID', 'LTeamID']]
    return build_matchups(team_stats, seeds_for_matchups, pool_results, config)


def build_loso_upset_matchup_pool(upset_config: dict, team_stats: pd.DataFrame, seeds_for_matchups: pd.DataFrame, tourney_results: pd.DataFrame) -> pd.DataFrame:
    '''builds the full favorite/underdog upset matchup pool over the training window, shared across every LOSO fold

    Parameters
    ----------
    upset_config : dict
        config of the upset experiment, to get the training season range and feature list
    team_stats : pd.DataFrame
        aggregated season stats per team per season
    seeds_for_matchups : pd.DataFrame
        seeds df with parsed integer seeds (see prepare_seed_columns)
    tourney_results : pd.DataFrame
        raw tournament results df from kaggle

    Returns
    -------
    pd.DataFrame
        upset matchup pool of the form season, features, label, covering every training season
    '''
    train_start, train_end = upset_config['data']['train_seasons']
    pool_results = tourney_results[
        (tourney_results['Season'] >= train_start) & (tourney_results['Season'] <= train_end)
    ][['Season', 'WTeamID', 'LTeamID']]
    return build_upset_matchups(team_stats, seeds_for_matchups, pool_results, upset_config)


def build_fold_config(config: dict, held_out_season: int) -> dict:
    '''deep copies a config and renames its experiment names so this fold's artifacts don't collide with others

    Parameters
    ----------
    config : dict
        base config to derive the fold config from
    held_out_season : int
        season being held out, folded into the experiment name for uniqueness

    Returns
    -------
    dict
        deep copy of config with data.experiment_name and model.experiment_name
        both set to '{original_experiment_name}_loso_{held_out_season}'
    '''
    fold_config = copy.deepcopy(config)
    fold_name = f'{config["model"]["experiment_name"]}_loso_{held_out_season}'
    fold_config['data']['experiment_name'] = fold_name
    fold_config['model']['experiment_name'] = fold_name
    return fold_config


def fold_arrays(matchups_pool: pd.DataFrame, feature_cols: list[str], fold_train_seasons: list[int], val_season: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''slices a shared matchup pool down to one fold's train and val seasons and converts to arrays

    Parameters
    ----------
    matchups_pool : pd.DataFrame
        full matchup pool, as returned by build_loso_matchup_pool / build_loso_upset_matchup_pool
    feature_cols : list[str]
        feature columns to extract, in order
    fold_train_seasons : list[int]
        seasons to include in this fold's training set
    val_season : int
        season to use as this fold's validation set

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train shape (N, F), y_train shape (N,), X_val shape (M, F), y_val shape (M,)
    '''
    train_df = matchups_pool[matchups_pool['Season'].isin(fold_train_seasons)]
    val_df = matchups_pool[matchups_pool['Season'] == val_season]
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df['label'].to_numpy()
    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df['label'].to_numpy()
    return X_train, y_train, X_val, y_val


def normalize_fold(fold_config: dict, X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    '''normalizes a fold's train/val features using only train-set statistics, and saves those stats to disk

    saving norm_stats per fold (rather than just returning it) means predict_win_prob
    functions built later against this fold's checkpoint can be reconstructed from disk
    alone, without needing this call's return value kept around

    Parameters
    ----------
    fold_config : dict
        this fold's config (see build_fold_config), for the save location
    X_train : np.ndarray
        this fold's train features shape (N, F)
    X_val : np.ndarray
        this fold's val features shape (M, F)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        normalized X_train shape (N, F), normalized X_val shape (M, F), and the norm stats
    '''
    # X_val is passed in as the third (would-be test) split since this fold has no test set of its own
    X_train_norm, X_val_norm, _, norm_stats = normalize_splits(X_train, X_val, X_val, fold_config)
    splits_dir = Path(fold_config['data']['splits_dir']) / fold_config['data']['experiment_name']
    splits_dir.mkdir(parents=True, exist_ok=True)
    path = splits_dir / 'norm_stats.json'
    with open(path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    return X_train_norm, X_val_norm, norm_stats


def fit_sklearn_fold(model_builder: Callable[[], Any], X_train: np.ndarray, y_train: np.ndarray) -> Any:
    '''builds a fresh sklearn-style model and fits it on this fold's training data

    Parameters
    ----------
    model_builder : Callable[[], Any]
        zero-arg factory returning an unfitted sklearn-style model
    X_train : np.ndarray
        normalized train features shape (N, F)
    y_train : np.ndarray
        train labels shape (N,)

    Returns
    -------
    Any
        the fitted model
    '''
    model = model_builder()
    model.fit(X_train, y_train)
    return model


def train_nn_fold(fold_config: dict, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> MatchupClassifier:
    '''builds a fresh MatchupClassifier and trains it on this fold's training data

    Parameters
    ----------
    fold_config : dict
        this fold's config (see build_fold_config), for model architecture and training hyperparams
    X_train : np.ndarray
        normalized train features shape (N, F)
    y_train : np.ndarray
        train labels shape (N,)
    X_val : np.ndarray
        normalized val features shape (M, F)
    y_val : np.ndarray
        val labels shape (M,)

    Returns
    -------
    MatchupClassifier
        the trained model (also checkpointed to disk by train_model)
    '''
    model = MatchupClassifier(fold_config)
    train_model(fold_config, model, X_train, y_train, X_val, y_val)
    return model


def run_loso_fold(configs: list[dict], held_out_season: int, all_seasons: list[int], team_stats: pd.DataFrame,
                   seeds: pd.DataFrame, matchups_pool: pd.DataFrame, tourney_results: pd.DataFrame,
                   slots_df: pd.DataFrame, needed_models: set[str], model_builders: Optional[dict[str, Callable[[], Any]]] = None,
                   n_sims: int = 1000, random_seed: int = 42, weights: dict = ROUND_WEIGHTS,
                   upset_config: Optional[dict] = None, upset_matchups_pool: Optional[pd.DataFrame] = None,
                   upset_threshold: float = 0.5) -> dict:
    '''trains and simulates every model still needed for one held-out season, then scores each against the actual bracket

    only builds/trains/predicts for the entries in needed_models, so a resumed
    backtest doesn't redo work for models this season already has results for

    Parameters
    ----------
    configs : list[dict]
        NN variant configs to evaluate, sharing a feature set (see validate_shared_feature_set)
    held_out_season : int
        season being held out as this fold's test season
    all_seasons : list[int]
        every season eligible for LOSO (see eligible_loso_seasons)
    team_stats : pd.DataFrame
        aggregated season stats per team per season
    seeds : pd.DataFrame
        raw or parsed seeds table (see build_seed_lookup)
    matchups_pool : pd.DataFrame
        full winner/loser matchup pool, as returned by build_loso_matchup_pool
    tourney_results : pd.DataFrame
        raw tournament results df from kaggle
    slots_df : pd.DataFrame
        raw slots df as returned by bracket.load_slots
    needed_models : set[str]
        model names still missing a result for this season
    model_builders : dict[str, Callable[[], Any]], optional
        model name -> zero-arg sklearn-style model factory, by default DEFAULT_MODEL_BUILDERS
    n_sims : int, optional
        number of bracket simulations to run per model, by default 1000
    random_seed : int, optional
        random seed shared across every model's simulations in this fold, so model
        comparisons aren't confounded by different random draws, by default 42
    weights : dict, optional
        round number -> ESPN-style points for a correct pick, by default ROUND_WEIGHTS
    upset_config : dict, optional
        config for the upset model backing the hybrid strategy, by default None (hybrid skipped)
    upset_matchups_pool : pd.DataFrame, optional
        full favorite/underdog matchup pool, required if upset_config is set, by default None
    upset_threshold : float, optional
        upset probability at or above which the hybrid strategy picks the underdog, by default 0.5

    Returns
    -------
    dict
        model name -> score_simulations summary (mean_score, std_score, scores) for this season
    '''
    model_builders = model_builders or DEFAULT_MODEL_BUILDERS
    base_config = configs[0]
    feature_cols = base_config['features']['columns']

    fold_train_seasons, val_season = fold_train_val_seasons(all_seasons, held_out_season)

    predict_fns = {}
    if 'chalk' in needed_models:
        predict_fns['chalk'] = chalk_predict_fn(seeds)
    if 'random' in needed_models:
        predict_fns['random'] = random_predict_fn()

    needed_sklearn_names = {name for name in model_builders if name in needed_models}
    nn_configs_needed = [
        cfg for cfg in configs
        if f'nn_{cfg["model"]["experiment_name"]}' in needed_models
    ]

    # only build/normalize the fold's arrays if something actually needs to train on them this season
    if needed_sklearn_names or nn_configs_needed:
        X_train, y_train, X_val, y_val = fold_arrays(matchups_pool, feature_cols, fold_train_seasons, val_season)
        base_fold_config = build_fold_config(base_config, held_out_season)
        X_train_norm, X_val_norm, norm_stats = normalize_fold(base_fold_config, X_train, X_val)

        if needed_sklearn_names:
            for name in needed_sklearn_names:
                model = fit_sklearn_fold(model_builders[name], X_train_norm, y_train)
                predict_fns[name] = classical_predict_fn(
                    model, team_stats, seeds, norm_stats, base_config
                )

        for cfg in nn_configs_needed:
            nn_fold_config = build_fold_config(cfg, held_out_season)
            train_nn_fold(nn_fold_config, X_train_norm, y_train, X_val_norm, y_val)
            model_label = f'nn_{cfg["model"]["experiment_name"]}'
            predict_fns[model_label] = nn_predict_fn(nn_fold_config, team_stats, seeds, norm_stats)

    hybrid_label = None
    if upset_config is not None:
        hybrid_label = f'hybrid_{upset_config["model"]["experiment_name"]}'

    if hybrid_label is not None and hybrid_label in needed_models:
        assert upset_config is not None and upset_matchups_pool is not None
        upset_feature_cols = upset_config['features']['columns']
        X_train_u, y_train_u, X_val_u, y_val_u = fold_arrays(
            upset_matchups_pool, upset_feature_cols, fold_train_seasons, val_season
        )
        upset_fold_config = build_fold_config(upset_config, held_out_season)
        X_train_u_norm, X_val_u_norm, upset_norm_stats = normalize_fold(upset_fold_config, X_train_u, X_val_u)

        train_nn_fold(upset_fold_config, X_train_u_norm, y_train_u, X_val_u_norm, y_val_u)
        upset_predict_fn = nn_predict_fn(upset_fold_config, team_stats, seeds, upset_norm_stats)
        predict_fns[hybrid_label] = hybrid_predict_fn(seeds, upset_predict_fn, upset_threshold)

    tree = build_bracket_tree(held_out_season, slots_df, seeds)
    actual_winners = resolve_actual_winners(held_out_season, tree, tourney_results)

    fold_results = {}
    for name, predict_fn in predict_fns.items():
        sims = simulate_bracket(held_out_season, tree, predict_fn, n_sims=n_sims, random_seed=random_seed, model_name=name)
        fold_results[name] = score_simulations(sims, actual_winners, tree, weights)

    return fold_results


def run_loso_backtest(configs: list[dict] | dict, team_stats: pd.DataFrame, seeds: pd.DataFrame, tourney_results: pd.DataFrame,
                       slots_df: pd.DataFrame, output_path: str | Path, model_builders: Optional[dict[str, Callable[[], Any]]] = None,
                       n_sims: int = 1000, random_seed: int = 42, weights: dict = ROUND_WEIGHTS, resume: bool = True,
                       upset_config: Optional[dict] = None, upset_threshold: float = 0.5) -> pd.DataFrame:
    '''runs a full leave-one-season-out backtest over every eligible season, appending results as each fold completes

    each held-out season is treated as a test set exactly once, with the model
    retrained from scratch on every other training season. results are appended to
    output_path fold by fold (rather than held in memory and written once at the end)
    so a long backtest can be resumed after an interruption via resume=True

    Parameters
    ----------
    configs : list[dict] or dict
        one or more NN variant configs to evaluate, sharing a feature set (see validate_shared_feature_set)
    team_stats : pd.DataFrame
        aggregated season stats per team per season
    seeds : pd.DataFrame
        raw seeds df from kaggle
    tourney_results : pd.DataFrame
        raw tournament results df from kaggle
    slots_df : pd.DataFrame
        raw slots df as returned by bracket.load_slots
    output_path : str or Path
        csv path results are appended to as each fold completes
    model_builders : dict[str, Callable[[], Any]], optional
        model name -> zero-arg sklearn-style model factory, by default DEFAULT_MODEL_BUILDERS
    n_sims : int, optional
        number of bracket simulations to run per model per season, by default 1000
    random_seed : int, optional
        random seed shared across every model's simulations, by default 42
    weights : dict, optional
        round number -> ESPN-style points for a correct pick, by default ROUND_WEIGHTS
    resume : bool, optional
        skip season/model combinations already present in output_path, by default True
    upset_config : dict, optional
        config for the upset model backing the hybrid strategy, by default None (hybrid skipped)
    upset_threshold : float, optional
        upset probability at or above which the hybrid strategy picks the underdog, by default 0.5

    Returns
    -------
    pd.DataFrame
        full accumulated results df (including any results loaded from a prior run)
    '''
    if isinstance(configs, dict):
        configs = [configs]
    validate_shared_feature_set(configs)
    for cfg in configs:
        check_loso_compatible(cfg)

    base_config = configs[0]
    seeds_for_matchups = prepare_seed_columns(seeds)
    matchups_pool = build_loso_matchup_pool(base_config, team_stats, seeds_for_matchups, tourney_results)

    upset_matchups_pool = None
    hybrid_keys = set()
    if upset_config is not None:
        check_loso_compatible(upset_config)
        upset_matchups_pool = build_loso_upset_matchup_pool(
            upset_config, team_stats, seeds_for_matchups, tourney_results
        )
        hybrid_keys = {f'hybrid_{upset_config["model"]["experiment_name"]}'}

    all_seasons = eligible_loso_seasons(base_config)
    sklearn_keys = set(model_builders or DEFAULT_MODEL_BUILDERS)
    nn_keys = {f'nn_{cfg["model"]["experiment_name"]}' for cfg in configs}
    expected_models = sklearn_keys | nn_keys | hybrid_keys | {'chalk', 'random'}

    existing_by_season = existing_models_by_season(output_path) if resume else {}

    pbar = tqdm(all_seasons, desc='LOSO backtest', unit='season', position=0, leave=True)
    for season in pbar:
        pbar.set_postfix({'season': season})

        # skip entirely if every expected model already has a result for this season
        needed_models = expected_models - existing_by_season.get(season, set())
        if not needed_models:
            continue

        if not season_has_bracket_data(season, slots_df):
            continue

        try:
            fold_results = run_loso_fold(
                configs, season, all_seasons, team_stats, seeds, matchups_pool,
                tourney_results, slots_df, needed_models, model_builders=model_builders,
                n_sims=n_sims, random_seed=random_seed, weights=weights,
                upset_config=upset_config, upset_matchups_pool=upset_matchups_pool,
                upset_threshold=upset_threshold,
            )
        except Exception:
            print(f'LOSO fold failed for season {season}, skipping')
            continue

        fold_rows = [
            {
                'season': season,
                'model': model_name,
                'mean_score': result['mean_score'],
                'std_score': result['std_score'],
                'scores': result['scores'],
            }
            for model_name, result in fold_results.items()
        ]
        _append_fold_rows(output_path, fold_rows)

    return load_loso_results(output_path)


def summarize_loso_results(results_df: pd.DataFrame) -> pd.DataFrame:
    '''summarizes per-season LOSO scores into per-model averages and a win rate against the chalk baseline

    Parameters
    ----------
    results_df : pd.DataFrame
        full results df as returned by run_loso_backtest / load_loso_results

    Returns
    -------
    pd.DataFrame
        one row per model: avg_mean_score, season_to_season_std, n_seasons, win_rate_vs_chalk
        (fraction of seasons where a model's mean_score beat chalk's, over seasons both have)
    '''
    summary = (
        results_df.groupby('model')['mean_score']
        .agg(['mean', 'std', 'count'])
        .rename(columns={'mean': 'avg_mean_score', 'std': 'season_to_season_std', 'count': 'n_seasons'})
        .sort_values('avg_mean_score', ascending=False)
    )

    chalk_by_season = results_df[results_df['model'] == 'chalk'].set_index('season')['mean_score']
    win_rates = {}
    for model_name in results_df['model'].unique():
        model_by_season = results_df[results_df['model'] == model_name].set_index('season')['mean_score']
        common_seasons = model_by_season.index.intersection(chalk_by_season.index)
        if len(common_seasons) == 0:
            win_rates[model_name] = float('nan')
            continue
        win_rates[model_name] = float((model_by_season[common_seasons] >= chalk_by_season[common_seasons]).mean())
    summary['win_rate_vs_chalk'] = pd.Series(win_rates)

    return summary


def paired_season_scores(results_df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    '''aligns two models' per-season mean_score on the seasons they both have results for

    Comparisons must be paired by season rather than by individual simulation draw --
    the entries in each row's 'scores' list all come from the same held-out season's
    bracket and are not independent of one another, so the season-level mean_score is
    the right unit for a paired test.

    Parameters
    ----------
    results_df : pd.DataFrame
        full results df as returned by run_loso_backtest / load_loso_results
    model_a : str
        first model name (as it appears in the 'model' column)
    model_b : str
        second model name

    Returns
    -------
    pd.DataFrame
        one row per common season, with columns season, model_a, model_b (mean_score
        for each), indexed by season
    '''
    a = results_df[results_df['model'] == model_a].set_index('season')['mean_score']
    b = results_df[results_df['model'] == model_b].set_index('season')['mean_score']
    common_seasons = a.index.intersection(b.index).sort_values()
    return pd.DataFrame({model_a: a[common_seasons], model_b: b[common_seasons]}, index=common_seasons)


def paired_comparison(results_df: pd.DataFrame, model_a: str, model_b: str) -> dict:
    '''runs a paired t-test and a Wilcoxon signed-rank test comparing two models' per-season mean_score

    Both tests are reported because n_seasons is typically small (one row per LOSO
    fold), which makes the t-test's normality assumption on the paired differences
    hard to trust on its own -- the Wilcoxon test serves as a nonparametric check
    that doesn't rely on that assumption.

    Parameters
    ----------
    results_df : pd.DataFrame
        full results df as returned by run_loso_backtest / load_loso_results
    model_a : str
        first model name
    model_b : str
        second model name

    Returns
    -------
    dict
        model_a, model_b, n_seasons, mean_diff (model_a - model_b), t_stat, t_pvalue,
        wilcoxon_stat, wilcoxon_pvalue. The wilcoxon fields are nan if n_seasons < 1
        after dropping zero differences (wilcoxon is undefined in that case), and
        every field but model_a/model_b/n_seasons is nan if n_seasons < 2

    Raises
    ------
    ValueError
        model_a and model_b have no common seasons in results_df
    '''
    paired = paired_season_scores(results_df, model_a, model_b)
    n_seasons = len(paired)
    if n_seasons == 0:
        raise ValueError(f'{model_a!r} and {model_b!r} have no common seasons in results_df -- cannot pair them.')

    diffs = paired[model_a] - paired[model_b]
    result = {'model_a': model_a, 'model_b': model_b, 'n_seasons': n_seasons, 'mean_diff': float(diffs.mean())}

    if n_seasons < 2:
        result.update(t_stat=float('nan'), t_pvalue=float('nan'), wilcoxon_stat=float('nan'), wilcoxon_pvalue=float('nan'))
        return result

    t_stat, t_pvalue = stats.ttest_rel(paired[model_a], paired[model_b])
    result['t_stat'] = float(t_stat)
    result['t_pvalue'] = float(t_pvalue)

    try:
        w_stat, w_pvalue = stats.wilcoxon(diffs)
        result['wilcoxon_stat'] = float(w_stat) # type: ignore
        result['wilcoxon_pvalue'] = float(w_pvalue) # type: ignore
    except ValueError:
        # every paired difference is exactly zero (or too few nonzero diffs remain) -- wilcoxon is undefined
        result['wilcoxon_stat'] = float('nan')
        result['wilcoxon_pvalue'] = float('nan')

    return result


def _holm_bonferroni(pvalues: list[float]) -> list[float]:
    '''applies the Holm-Bonferroni step-down correction to a list of p-values

    Preferred over a plain Bonferroni correction here since it controls the same
    family-wise error rate while being uniformly more powerful (fewer false negatives).

    Parameters
    ----------
    pvalues : list[float]
        raw p-values, one per hypothesis test in the family

    Returns
    -------
    list[float]
        adjusted p-values in the same order as pvalues, each clipped to at most 1.0
    '''
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * pvalues[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def pairwise_comparison_table(results_df: pd.DataFrame, models: Optional[list[str]] = None,
                               reference: Optional[str] = None) -> pd.DataFrame:
    '''builds a table of paired t-test / Wilcoxon comparisons across models, with a multiple-comparison correction

    Parameters
    ----------
    results_df : pd.DataFrame
        full results df as returned by run_loso_backtest / load_loso_results
    models : list[str], optional
        models to compare, by default every model present in results_df
    reference : str, optional
        if given, only compare each other model against this one (e.g. 'chalk'),
        by default None (compare every pair among models)

    Returns
    -------
    pd.DataFrame
        one row per comparison: model_a, model_b, n_seasons, mean_diff, t_stat,
        t_pvalue, t_pvalue_holm, wilcoxon_stat, wilcoxon_pvalue, wilcoxon_pvalue_holm.
        The *_holm columns are Holm-Bonferroni-adjusted across all rows in the table,
        since running many pairwise tests inflates the chance of a false positive
        somewhere in the table if read off the raw p-values
    '''
    models = models if models is not None else sorted(results_df['model'].unique())
    if reference is not None:
        pairs = [(reference, m) for m in models if m != reference]
    else:
        pairs = list(combinations(models, 2))

    rows = [paired_comparison(results_df, model_a, model_b) for model_a, model_b in pairs]
    table = pd.DataFrame(rows)
    table['t_pvalue_holm'] = _holm_bonferroni(table['t_pvalue'].tolist())
    table['wilcoxon_pvalue_holm'] = _holm_bonferroni(table['wilcoxon_pvalue'].tolist())
    return table[[
        'model_a', 'model_b', 'n_seasons', 'mean_diff',
        't_stat', 't_pvalue', 't_pvalue_holm',
        'wilcoxon_stat', 'wilcoxon_pvalue', 'wilcoxon_pvalue_holm',
    ]]

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def plot_pvalue_heatmap(table: pd.DataFrame, models: Optional[list[str]] = None,
                         pvalue_col: str = 'wilcoxon_pvalue_holm', alpha: float = 0.05,
                         title: Optional[str] = None, ax=None):
    '''plots a symmetric heatmap of pairwise p-values from pairwise_comparison_table

    Parameters
    ----------
    table : pd.DataFrame
        output of pairwise_comparison_table
    models : list[str], optional
        order/subset of models to include, by default every model appearing in table
    pvalue_col : str
        which column to plot, by default the Holm-adjusted Wilcoxon p-value
    alpha : float
        significance threshold used to annotate significant cells with an asterisk
    title : str, optional
        plot title
    ax : matplotlib.axes.Axes, optional
        axes to draw on; a new figure is created if not given

    Returns
    -------
    matplotlib.axes.Axes
    '''
    if models is None:
        models = sorted(set(table['model_a']) | set(table['model_b']))
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    mat = np.full((n, n), np.nan)
    for _, row in table.iterrows():
        if row['model_a'] in idx and row['model_b'] in idx:
            i, j = idx[row['model_a']], idx[row['model_b']]
            mat[i, j] = row[pvalue_col]
            mat[j, i] = row[pvalue_col]

    annot = np.full((n, n), '', dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = ''
            elif np.isnan(mat[i, j]):
                annot[i, j] = ''
            else:
                marker = '*' if mat[i, j] < alpha else ''
                annot[i, j] = f'{mat[i, j]:.3f}{marker}'

    if ax is None:
        fig, ax = plt.subplots(figsize=(0.9 * n + 2, 0.9 * n + 2))

    # diverging norm centered at alpha so significant cells pop visually
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=alpha, vmax=1)
    mask = np.eye(n, dtype=bool)

    sns.heatmap(
        mat, mask=mask, annot=annot, fmt='', cmap='RdYlGn', norm=norm,
        xticklabels=models, yticklabels=models, square=True, linewidths=0.5,
        cbar_kws={'label': pvalue_col}, ax=ax,
    )
    ax.set_title(title or f'{pvalue_col} (Holm-adjusted, * = p < {alpha})')
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    return ax