import re
from pathlib import Path
from typing import Callable, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from src.model.inference import get_nn_predictions

# espn round weighting * 10 for cleaner numbers
ROUND_WEIGHTS = {
    1: 10,
    2: 20,
    3: 40,
    4: 80,
    5: 160,
    6: 320,
}


def is_seed_code(code: str) -> bool:
    '''checks whether a string looks like a raw seed code (e.g. 'W16'), not whether it actually is one

    a seed code is a region letter followed by a two digit seed number. this is a shape
    check only, see build_bracket_tree's resolve_ref for why shape alone isn't enough
    to tell a seed code from a First Four slot reference

    Parameters
    ----------
    code : str
        candidate string to check

    Returns
    -------
    bool
        True if code matches the WXYZ + two digit pattern
    '''
    return bool(re.compile(r'^[WXYZ]\d{2}$').match(code))


def is_slot_ref(value: str | int) -> bool:
    '''checks whether a bracket tree value is a reference to another slot or a resolved team id

    slot refs are strings (slot names like 'R1W1'), resolved teams are integer team ids

    Parameters
    ----------
    value : str or int
        value to check, either a slot name or a team id

    Returns
    -------
    bool
        True if value is a slot reference (str), False if it's a team id
    '''
    return isinstance(value, str)


def slot_round(slot: str) -> int:
    '''parses the round number out of a slot name

    round 1-6 for normal slots (Rn... naming). round 0 for play-in
    slots, which are named directly after the seed they resolve (W16, W16a, W16b)
    rather than with an 'Rn' prefix (R1W1, W1, W16)

    Parameters
    ----------
    slot : str
        slot name to parse

    Returns
    -------
    int
        round number, 0-6

    Raises
    ------
    ValueError
        the slot name doesn't match either the 'Rn...' or seed code pattern
    '''
    match = re.match(r'^R(\d)', slot)
    if match:
        return int(match.group(1))
    if re.match(r'^[WXYZ]\d{2}$', slot):
        return 0
    raise ValueError(f'could not parse round number out of slot {slot!r}')


def load_slots(raw_dir: str) -> pd.DataFrame:
    '''loads the raw kaggle bracket slot structure csv

    Parameters
    ----------
    raw_dir : str
        directory containing the raw kaggle csvs

    Returns
    -------
    pd.DataFrame
        raw slots df of the form Season, Slot, StrongSeed, WeakSeed
    '''
    return pd.read_csv(Path(raw_dir) / 'MNCAATourneySlots.csv')


def build_bracket_tree(season: int, slots_df: pd.DataFrame, seeds_df: pd.DataFrame) -> dict:
    '''builds the bracket structure for one season as a tree of slots

    a ref is either a team_id (leaf, this side of the matchup is a known team)
    or another slot string (this side is 'whoever wins slot X'). leaf-ness is
    decided by presence in this season's seeds table, not by string shape bc
    play-in seasons have slot codes like 'W16' that look like a seed code
    but are actually a reference to the play-in game's own slot (whose Slot ID
    is that same string, e.g. Slot='W16', StrongSeed='W16a', WeakSeed='W16b')

    Parameters
    ----------
    season : int
        season to build the tree for
    slots_df : pd.DataFrame
        raw slots df as returned by load_slots
    seeds_df : pd.DataFrame
        seeds table, raw or already parsed (see build_seed_lookup)

    Returns
    -------
    dict
        {slot: {'strong': ref, 'weak': ref, 'round': int}} for one season
    '''
    season_slots = slots_df[slots_df['Season'] == season]
    season_seeds = seeds_df[seeds_df['Season'] == season]
    seed_to_team = dict(zip(season_seeds['Seed'], season_seeds['TeamID']))

    valid_slots = set(season_slots['Slot'])

    def resolve_ref(slot, side_name, code):
        # this season's own slot names take priority: a code that matches a real slot
        # even one that looks like a seed code 'W16' is a play-in reference
        if code in valid_slots:
            return code
        # otherwise, if it's a known seed code for this season, resolve straight to the team
        if code in seed_to_team:
            return seed_to_team[code]
        # code looks like a seed but isn't a real seed or a real slot this season then bad data
        if is_seed_code(code):
            raise KeyError(f'season {season} slot {slot}: {side_name}={code!r} looks like a seed but isnt seed or slot')
        # anything else is already a genuine slot reference like 'R1W1', pass through as-is
        return code


    tree = {}
    for row in season_slots.itertuples():
        slot, strong, weak = str(row.Slot), row.StrongSeed, row.WeakSeed

        strong_ref = resolve_ref(slot, 'StrongSeed', strong)
        weak_ref = resolve_ref(slot, 'WeakSeed', weak)

        tree[slot] = {
            'strong': strong_ref,
            'weak': weak_ref,
            'round': slot_round(slot),
        }
    return tree


def final_slot(tree: dict) -> str:
    '''finds the championship slot, the one with the highest round number

    Parameters
    ----------
    tree : dict
        bracket tree as built by build_bracket_tree

    Returns
    -------
    str
        slot name of the championship game
    '''
    return max(tree, key=lambda s: tree[s]['round'])


def build_seed_lookup(seed_df: pd.DataFrame) -> pd.Series:
    '''builds a (Season, TeamID) -> int seed lookup, parsing seeds if needed

    accepts either the raw seeds table (Season, TeamID, Seed as e.g. 'W01')
    or one that already has a numeric 'seed' column (as produced by
    dataset.parse_seed)

    Parameters
    ----------
    seed_df : pd.DataFrame
        seeds table, raw or already parsed

    Returns
    -------
    pd.Series
        seed lookup indexed by (Season, TeamID)
    '''
    if 'seed' not in seed_df.columns:
        from src.data.dataset import parse_seed
        seed_df = seed_df.copy()
        seed_df['seed'] = seed_df['Seed'].apply(parse_seed)
    return seed_df.set_index(['Season', 'TeamID'])['seed']


def chalk_predict_fn(seed_df: pd.DataFrame) -> Callable:
    '''builds a predict_win_prob function that always picks the lower seed

    Parameters
    ----------
    seed_df : pd.DataFrame
        seeds table, raw or already parsed (see build_seed_lookup)

    Returns
    -------
    Callable
        predict_win_prob(team_a, team_b, season) -> float, 1.0 if team_a is the
        better seed, 0.0 otherwise (including ties, since there's no seed edge to pick)
    '''
    seed_lookup = build_seed_lookup(seed_df)

    def predict_win_prob(team_a, team_b, season):
        seed_a = seed_lookup.loc[(season, team_a)]
        seed_b = seed_lookup.loc[(season, team_b)]
        return 1.0 if seed_a < seed_b else 0.0

    return predict_win_prob


def random_predict_fn() -> Callable:
    '''builds a predict_win_prob function that flips a coin every game so that repeated samples are 50/50

    Returns
    -------
    Callable
        predict_win_prob(team_a, team_b, season) -> float, always 0.5
    '''
    def predict_win_prob(team_a, team_b, season):
        return 0.5

    return predict_win_prob


def make_feature_row(team_a: int, team_b: int, season: int, stats_lookup: pd.DataFrame, seed_lookup: pd.Series, config: dict) -> np.ndarray:
    '''builds a single team_a - team_b difference feature row, mirroring build_matchups at inference time

    Parameters
    ----------
    team_a : int
        first team id
    team_b : int
        second team id
    season : int
        season the matchup is in
    stats_lookup : pd.DataFrame
        team stats indexed by (season, team_id)
    seed_lookup : pd.Series
        seeds indexed by (season, team_id)
    config : dict
        config of the experiment, to build the correct features

    Returns
    -------
    np.ndarray
        unnormalized feature row shape (F,), in the order given by config['features']['columns']
    '''
    feature_cols = config['features']['columns']
    diff_features = [f for f in feature_cols if f != 'season_norm']

    stats_a = stats_lookup.loc[(season, team_a), :]
    stats_b = stats_lookup.loc[(season, team_b), :]

    row = {}
    for feature in diff_features:
        col = feature.replace('_diff', '')
        # seed isn't in the team stats table, so it needs its own lookup rather than stats_a/stats_b
        if col == 'seed':
            row[feature] = float(seed_lookup.loc[(season, team_a)]) - float(seed_lookup.loc[(season, team_b)])
        else:
            row[feature] = float(stats_a[col]) - float(stats_b[col])

    if 'season_norm' in feature_cols:
        min_season = config['data']['train_seasons'][0]
        max_season = config['data']['test_season']
        row['season_norm'] = (season - min_season) / (max_season - min_season)

    return np.array([row[f] for f in feature_cols])


def classical_predict_fn(model, team_stats: pd.DataFrame, seed_df: pd.DataFrame, norm_stats: dict, config: dict) -> Callable:
    '''wraps a fitted sklearn-style model (predict_proba) for use as predict_win_prob

    Parameters
    ----------
    model
        fitted sklearn-style model like lr or rf with a predict_proba method
    team_stats : pd.DataFrame
        aggregated season stats per team per season
    seed_df : pd.DataFrame
        seeds table, raw or already parsed (see build_seed_lookup)
    norm_stats : dict
        normalization stats (mean, std) saved from training, to match features at inference
    config : dict
        config of the experiment, to build the correct features

    Returns
    -------
    Callable
        predict_win_prob(team_a, team_b, season) -> float win probability for team_a
    '''
    stats_lookup = team_stats.set_index(['season', 'team_id'])
    seed_lookup = build_seed_lookup(seed_df)
    mean = np.array(norm_stats['mean'])
    std = np.array(norm_stats['std'])

    def predict_win_prob(team_a, team_b, season):
        x = make_feature_row(team_a, team_b, season, stats_lookup, seed_lookup, config)
        # normalize with the saved train-set stats, same as normalize_splits at train time
        x_norm = ((x - mean) / std).reshape(1, -1)
        return float(model.predict_proba(x_norm)[0, 1])

    return predict_win_prob


def nn_predict_fn(config: dict, team_stats: pd.DataFrame, seed_df: pd.DataFrame, norm_stats: dict) -> Callable:
    '''wraps get_nn_predictions for use as predict_win_prob

    Parameters
    ----------
    config : dict
        config of the experiment, to build the correct features and locate the checkpoint
    team_stats : pd.DataFrame
        aggregated season stats per team per season
    seed_df : pd.DataFrame
        seeds table, raw or already parsed (see build_seed_lookup)
    norm_stats : dict
        normalization stats (mean, std) saved from training, to match features at inference

    Returns
    -------
    Callable
        predict_win_prob(team_a, team_b, season) -> float win probability for team_a
    '''

    stats_lookup = team_stats.set_index(['season', 'team_id'])
    seed_lookup = build_seed_lookup(seed_df)
    mean = np.array(norm_stats['mean'])
    std = np.array(norm_stats['std'])

    def predict_win_prob(team_a, team_b, season):
        x = make_feature_row(team_a, team_b, season, stats_lookup, seed_lookup, config)
        x_norm = ((x - mean) / std).reshape(1, -1)
        probs = get_nn_predictions(config, x_norm)
        return float(probs[0])

    return predict_win_prob


def hybrid_predict_fn(seed_df: pd.DataFrame, upset_predict_fn: Callable, threshold: float) -> Callable:
    '''chalk by default, overridden by an upset call when the upset model is confident enough

    upset_predict_fn is any predict_win_prob-style callable like the output of
    classical_predict_fn or nn_predict_fn built against an upset data model
    but it must be called here in (favorite, underdog) order, since that's the
    orientation the upset model was trained on

    ties (equal seed, no defined favorite) fall back to chalk's own tie
    behavior rather than calling the upset model at all bc no
    favorite/underdog to orient the feature row around

    Parameters
    ----------
    seed_df : pd.DataFrame
        seeds table, raw or already parsed (build_seed_lookup)
    upset_predict_fn : Callable
        predict_win_prob-style callable for the upset model, called as (favorite, underdog, season)
    threshold : float
        upset probability at or above which the underdog is picked to win

    Returns
    -------
    Callable
        predict_win_prob(team_a, team_b, season) -> float win probability for team_a
    '''
    seed_lookup = build_seed_lookup(seed_df)

    def predict_win_prob(team_a, team_b, season):
        seed_a = seed_lookup.loc[(season, team_a)]
        seed_b = seed_lookup.loc[(season, team_b)]

        if seed_a == seed_b:
            return 0.0  # matches chalk_predict_fn's tie behavior

        # orient favorite/underdog by seed so the upset model sees the orientation it was trained on
        favorite, underdog = (team_a, team_b) if seed_a < seed_b else (team_b, team_a)
        upset_prob = upset_predict_fn(favorite, underdog, season)

        # translate the favorite/underdog outcome back into a team_a/team_b win prob
        favorite_is_a = (favorite == team_a)
        if upset_prob >= threshold:
            return 0.0 if favorite_is_a else 1.0
        return 1.0 if favorite_is_a else 0.0

    return predict_win_prob


def resolve_slot(slot: str, tree: dict, predict_win_prob: Callable, season: int, rng: np.random.Generator, cache: dict) -> int:
    '''recursively resolves a bracket slot to a winning team id by simulating every game feeding into it

    walks down to leaf teams first via recursion on slot refs, then simulates each
    game on the way back up using predict_win_prob and rng, caching results per slot
    so a slot feeding into multiple later games is only simulated once

    Parameters
    ----------
    slot : str
        slot to resolve
    tree : dict
        bracket tree as built by build_bracket_tree
    predict_win_prob : Callable
        predict_win_prob(team_a, team_b, season) -> float win probability for team_a
    season : int
        season being simulated
    rng : np.random.Generator
        random generator to draw simulated outcomes from
    cache : dict
        slot -> winning team id, mutated in place as slots are resolved

    Returns
    -------
    int
        winning team id for this slot
    '''
    if slot in cache:
        return cache[slot]

    node = tree[slot]
    strong = node['strong']
    weak = node['weak']

    # recurse into any side that's itself a slot reference rather than an already-known team
    team_a = strong if not is_slot_ref(strong) else resolve_slot(strong, tree, predict_win_prob, season, rng, cache)
    team_b = weak if not is_slot_ref(weak) else resolve_slot(weak, tree, predict_win_prob, season, rng, cache)

    prob_a_wins = predict_win_prob(team_a, team_b, season)
    winner = team_a if rng.random() < prob_a_wins else team_b

    cache[slot] = winner
    return winner


def simulate_bracket_once(season: int, tree: dict, predict_win_prob: Callable, rng: np.random.Generator) -> dict:
    '''simulates one full bracket by resolving the championship slot which recursively resolves everything below it

    Parameters
    ----------
    season : int
        season being simulated
    tree : dict
        bracket tree as built by build_bracket_tree
    predict_win_prob : Callable
        predict_win_prob(team_a, team_b, season) -> float win probability for team_a
    rng : np.random.Generator
        random generator to draw simulated outcomes from

    Returns
    -------
    dict
        slot -> winning team id for every slot in the bracket
    '''
    cache = {}
    resolve_slot(final_slot(tree), tree, predict_win_prob, season, rng, cache)
    return cache


def simulate_bracket(season: int, tree: dict, predict_win_prob: Callable, n_sims: int = 1000, random_seed: Optional[int] = None, model_name: Optional[str] = None) -> list[dict]:
    '''runs n_sims independent simulations of the full bracket for one season

    model_name is purely cosmetic -- shown in the progress bar description so
    you can tell which model's simulation is running (e.g. when this is
    called once per model inside a loop). position=1 keeps this bar on its
    own line below any outer/caller-level tqdm bar (e.g. a season-level bar
    in loso.py) instead of both bars defaulting to position 0 and overwriting
    each other's line every time this one is recreated

    Parameters
    ----------
    season : int
        season being simulated
    tree : dict
        bracket tree as built by build_bracket_tree
    predict_win_prob : Callable
        predict_win_prob(team_a, team_b, season) -> float win probability for team_a
    n_sims : int, optional
        number of independent simulations to run, by default 1000
    random_seed : int, optional
        seed for the random generator, by default None
    model_name : str, optional
        cosmetic label shown in the progress bar description, by default None

    Returns
    -------
    list[dict]
        list of n_sims dicts, each {slot: winning_team_id} for one simulated bracket
    '''
    rng = np.random.default_rng(random_seed)
    desc = f'simulating bracket ({model_name})' if model_name else 'simulating bracket'
    return [
        simulate_bracket_once(season, tree, predict_win_prob, rng)
        for _ in tqdm(range(n_sims), desc=desc, unit='sims', leave=False, position=1)
    ]


def slot_winner_distribution(all_sims: list[dict], slot: str) -> dict:
    '''computes how often each team won a given slot across simulations

    Parameters
    ----------
    all_sims : list[dict]
        list of simulated brackets, each slot -> winning team id, as returned by simulate_bracket
    slot : str
        slot to compute the winner distribution for

    Returns
    -------
    dict
        team_id -> fraction of simulations that team won this slot
    '''
    winners = [sim[slot] for sim in all_sims]
    values, counts = np.unique(winners, return_counts=True)
    return dict(zip(values.tolist(), (counts / len(all_sims)).tolist()))


def resolve_actual_slot(slot: str, tree: dict, game_winner: dict, cache: dict) -> int:
    '''recursively resolves a bracket slot to the actual winning team id, using real recorded results

    same recursive shape as resolve_slot, but instead of simulating each game with
    predict_win_prob, it looks up the real winner from game_winner built by
    resolve_actual_winners from tourney_results

    Parameters
    ----------
    slot : str
        slot to resolve
    tree : dict
        bracket tree as built by build_bracket_tree
    game_winner : dict
        frozenset({team_a, team_b}) -> winning team id for every actually played game
    cache : dict
        slot -> actual winning team id, mutated in place as slots are resolved

    Returns
    -------
    int
        actual winning team id for this slot

    Raises
    ------
    ValueError
        no recorded game exists between the two teams expected to meet at this slot
    '''
    if slot in cache:
        return cache[slot]

    node = tree[slot]
    strong = node['strong']
    weak = node['weak']

    team_a = strong if not is_slot_ref(strong) else resolve_actual_slot(strong, tree, game_winner, cache)
    team_b = weak if not is_slot_ref(weak) else resolve_actual_slot(weak, tree, game_winner, cache)

    # order doesn't matter for who actually played whom, so look the pair up as a frozenset
    pair = frozenset((team_a, team_b))
    if pair not in game_winner:
        raise ValueError(f'no recorded game between teams {team_a} and {team_b} for slot {slot}')

    cache[slot] = game_winner[pair]
    return cache[slot]


def resolve_actual_winners(season: int, tree: dict, tourney_results: pd.DataFrame) -> dict:
    '''resolves every slot to its actual winning team using the same tree traversal as the simulator

    grounded in real results instead of predict_win_prob

    Parameters
    ----------
    season : int
        season to resolve actual winners for
    tree : dict
        bracket tree as built by build_bracket_tree
    tourney_results : pd.DataFrame
        tournament results df formatted by season and WTeamID LTeamID

    Returns
    -------
    dict
        slot -> actual winning team id for every slot in the bracket
    '''
    season_games = tourney_results[tourney_results['Season'] == season]
    game_winner = {
        frozenset((row.WTeamID, row.LTeamID)): row.WTeamID
        for row in season_games.itertuples()
    }

    cache = {}
    resolve_actual_slot(final_slot(tree), tree, game_winner, cache)
    return cache


def espn_score(sim_winners: dict, actual_winners: dict, tree: dict, weights: dict = ROUND_WEIGHTS) -> int:
    '''scores one simulated bracket against the actual results using ESPN-style round weighting

    Parameters
    ----------
    sim_winners : dict
        slot -> winning team id for one simulated bracket
    actual_winners : dict
        slot -> actual winning team id, as returned by resolve_actual_winners
    tree : dict
        bracket tree as built by build_bracket_tree, to look up each slot's round
    weights : dict, optional
        round number -> points awarded for a correct pick in that round, by default ROUND_WEIGHTS

    Returns
    -------
    int
        total ESPN-style score for this simulated bracket
    '''
    # round 0 isnt scored
    score = 0
    for slot, winner in sim_winners.items():
        if winner == actual_winners.get(slot):
            score += weights.get(tree[slot]['round'], 0)
    return score


def score_simulations(all_sims: list[dict], actual_winners: dict, tree: dict, weights: dict = ROUND_WEIGHTS) -> dict:
    '''scores every simulated bracket against the actual results and summarizes across simulations

    Parameters
    ----------
    all_sims : list[dict]
        list of simulated brackets, each slot -> winning team id, as returned by simulate_bracket
    actual_winners : dict
        slot -> actual winning team id, as returned by resolve_actual_winners
    tree : dict
        bracket tree as built by build_bracket_tree
    weights : dict, optional
        round number -> points awarded for a correct pick in that round, by default ROUND_WEIGHTS

    Returns
    -------
    dict
        mean and std ESPN-style score across all simulations, plus the raw per-sim scores
    '''
    scores = [espn_score(sim, actual_winners, tree, weights) for sim in all_sims]
    return {
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'scores': scores,
    }