from typing import Callable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from src.simulation.bracket import is_slot_ref, final_slot, slot_winner_distribution

def plot_score_distributions(results_df: pd.DataFrame, models: Optional[list[str]] = None, ax: Optional[Axes] = None) -> Axes:
    '''boxplots the per simulation espn score distribution for each model pooled across all LOSO seasons

    Parameters
    ----------
    results_df : pd.DataFrame
        LOSO results df as returned by loso.load_loso_results, with a 'scores' column of per sim score lists
    models : list[str], optional
        models to plot, by default None for every model in results_df, sorted
    ax : Axes, optional
        axes to draw on, by default None for a new figure

    Returns
    -------
    Axes
        the axes the boxplot was drawn on
    '''
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    if models is None:
        models = sorted(results_df['model'].unique())
    # pool every season's per-sim scores together for each model into one flat array
    data = [np.concatenate(results_df.loc[results_df['model'] == m, 'scores'].values) for m in models]
    ax.boxplot(data, tick_labels=models, showmeans=True)
    ax.set_ylabel('ESPN score')
    ax.set_title('Per-simulation score distribution by model (all LOSO seasons pooled)')
    ax.tick_params(axis='x', rotation=30)
    return ax


def plot_score_trend(results_df: pd.DataFrame, models: Optional[list[str]] = None, ax: Optional[Axes] = None) -> Axes:
    '''line plots each model's mean LOSO score by held out season

    Parameters
    ----------
    results_df : pd.DataFrame
        LOSO results df as returned by loso.load_loso_results
    models : list[str], optional
        models to plot, by default None for every model in results_df, sorted
    ax : Axes, optional
        axes to draw on, by default None for a new figure

    Returns
    -------
    Axes
        the axes the trend lines were drawn on
    '''
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
        
    if models is None:
        models = sorted(results_df['model'].unique())
        
    for model in models:
        sub = results_df[results_df['model'] == model].sort_values('season')
        ax.plot(sub['season'], sub['mean_score'], marker='o', label=model)
        
    min_season = int(results_df['season'].min())
    max_season = int(results_df['season'].max())

    ax.set_xticks(range(min_season, max_season + 1, 2))
    ax.set_xlabel('held-out season')
    ax.set_ylabel('mean ESPN score')
    ax.set_title('LOSO mean score by season')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    
    return ax


def plot_win_rate_vs_chalk(results_df: pd.DataFrame, benchmark_model: str = 'chalk', models: Optional[list[str]] = None, ax: Optional[Axes] = None) -> Axes:
    '''bar plots the fraction of seasons each model tied or beat the chalk model's mean score

    Parameters
    ----------
    results_df : pd.DataFrame
        LOSO results df as returned by loso.load_loso_results
    benchmark_model : str, optional
        model name to use as the reference baseline, by default 'chalk'
    models : list[str], optional
        models to compare against chalk_model, by default None for every other model in results_df, sorted
    ax : Axes, optional
        axes to draw on, by default None for a new figure

    Returns
    -------
    Axes
        the axes the bar chart was drawn on
    '''
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    chalk = results_df[results_df['model'] == benchmark_model].set_index('season')['mean_score']
    if models is None:
        models = sorted(m for m in results_df['model'].unique() if m != benchmark_model)
    win_rates = {}
    for model in models:
        sub = results_df[results_df['model'] == model].set_index('season')['mean_score']
        # only compare on seasons both models actually have a result for
        common_seasons = sub.index.intersection(chalk.index)
        if len(common_seasons) == 0:
            continue
        win_rates[model] = float((sub.loc[common_seasons] >= chalk.loc[common_seasons]).mean())
    ax.bar(list(win_rates.keys()), list(win_rates.values()))
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f'fraction of seasons beating {benchmark_model}')
    ax.set_title(f'Win rate vs {benchmark_model}')
    ax.tick_params(axis='x', rotation=30)
    return ax


def plot_score_heatmap(results_df: pd.DataFrame, models: Optional[list[str]] = None, ax: Optional[Axes] = None) -> Axes:
    '''heatmaps mean LOSO score by season x model, with the value annotated in each cell

    Parameters
    ----------
    results_df : pd.DataFrame
        LOSO results df as returned by loso.load_loso_results
    models : list[str], optional
        models (and column order) to include, by default None for every model in results_df
    ax : Axes, optional
        axes to draw on, by default None for a new figure, sized to fit grid

    Returns
    -------
    Axes
        the axes the heatmap was drawn on
    '''
    pivot = results_df.pivot(index='season', columns='model', values='mean_score')
    if models is not None:
        pivot = pivot[models]
    if ax is None:
        _, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 2, 0.5 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title('Mean ESPN score by season x model')
    plt.colorbar(im, ax=ax, label='mean score')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                # flip text color to white on dark cells so the annotation stays legible
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                         color='white' if val < np.nanmax(pivot.values) * 0.6 else 'black', fontsize=8)
    return ax


def _team_label(team_id: int, team_names: Optional[dict], seeds: Optional[dict] = None, is_leaf: bool = False) -> str:
    '''builds the display label for one team, optionally prefixed with its seed

    Parameters
    ----------
    team_id : int
        team id to label
    team_names : dict, optional
        team_id -> display name lookup, by default None (falls back to the raw team_id)
    seeds : dict, optional
        team_id -> seed lookup, only used when is_leaf is True, by default None
    is_leaf : bool, optional
        whether this team is a first-round entrant (leaf of the tree), by default False

    Returns
    -------
    str
        the team's display label, with its seed prefixed if is_leaf and a seed is available
    '''
    name = str(team_id)
    if team_names is not None:
        name = team_names.get(team_id, name)
    # only leaves get a seed prefix -- teams past round 1 are already identified by name alone
    if is_leaf and seeds and team_id in seeds:
        name = f"{seeds[team_id]} {name}"
    return name


def _walk_layout(tree: dict, label_fn: Callable) -> tuple[dict, dict]:
    '''computes x/y plot coordinates for every node in the bracket tree via a depth-first walk

    x position comes directly from each slot's round number (leaves sit one column to the
    left of the slot they feed into). y position is assigned to leaves in visitation order,
    then each parent slot is placed at the midpoint of its two children's y positions, so
    the whole tree lays out as a standard bracket diagram

    Parameters
    ----------
    tree : dict
        bracket tree as built by bracket.build_bracket_tree
    label_fn : Callable
        label_fn(slot, team_leaf) -> (team, extra) -- called with slot=None for leaves
        (team_leaf is the leaf's team id) and with team_leaf=None for internal slots
        (slot is the slot name); returns the team id to display and an optional extra
        annotation string

    Returns
    -------
    tuple[dict, dict]
        x_pos (slot -> x coordinate, from its round number) and node_xy
        ((slot_or_parent, side) -> (x, y, team, extra, source_slot) for every drawn edge endpoint)
    '''
    x_pos = {slot: node['round'] for slot, node in tree.items()}
    node_xy = {}
    y_counter = [0]

    def walk(ref, parent_slot, side):
        if not is_slot_ref(ref):
            # leaf: a real team id, not a slot to recurse into. assign it the next free y slot
            y = y_counter[0]
            y_counter[0] += 1
            x = x_pos[parent_slot] - 1
            team, extra = label_fn(None, ref)
            # Store (x, y, team, extra, source_slot)
            node_xy[(parent_slot, side)] = (x, y, team, extra, None)
            return y
        
        slot = ref
        node = tree[slot]
        y_s = walk(node['strong'], slot, 'strong')
        y_w = walk(node['weak'], slot, 'weak')
        y = (y_s + y_w) / 2

        team, extra = label_fn(slot, None)
        node_xy[(slot, None)] = (x_pos[slot], y, team, extra, slot)
        if parent_slot is not None:
            node_xy[(parent_slot, side)] = (x_pos[slot], y, team, extra, slot)
        return y

    walk(final_slot(tree), None, None)
    return x_pos, node_xy


def _draw_tree(tree: dict, x_pos: dict, node_xy: dict, team_names: Optional[dict], title: str, ax: Axes, slot_color_fn: Callable, seeds: Optional[dict] = None) -> None:
    '''draws the bracket lines, labels, and champion callout for a laid-out tree

    Parameters
    ----------
    tree : dict
        bracket tree as built by bracket.build_bracket_tree
    x_pos : dict
        slot -> x coordinate, as returned by _walk_layout
    node_xy : dict
        (slot_or_parent, side) -> (x, y, team, extra, source_slot), as returned by _walk_layout
    team_names : dict, optional
        team_id -> display name lookup, passed through to _team_label
    title : str
        title to set on the axes
    ax : Axes
        axes to draw on
    slot_color_fn : Callable
        slot_color_fn(source_slot, team) -> matplotlib color string, used to color each label
    seeds : dict, optional
        team_id -> seed lookup, passed through to _team_label
    '''
    for slot, node in tree.items():
        for side in ('strong', 'weak'):
            cx, cy, c_team, c_extra, c_source_slot = node_xy[(slot, side)]
            sx = x_pos[slot]
            
            # Draw horizontal line
            ax.plot([cx, sx], [cy, cy], color='#999999', linewidth=1, zorder=1)
            
            is_leaf = (c_source_slot is None)
            label = _team_label(c_team, team_names, seeds, is_leaf) if c_team is not None else '?'
            
            if c_extra:
                label = f'{label} ({c_extra})'
            
            # First round entrants default to black; otherwise calculate color based on the slot they won
            color = '#000000' if is_leaf else slot_color_fn(c_source_slot, c_team)
            
            # Rest label directly on top of the line, starting at the left side
            ax.text(cx + 0.05, cy + 0.05, label, ha='left', va='bottom', fontsize=7, color=color, zorder=2)
            
        # Draw vertical line connecting the two children
        ax.plot([x_pos[slot], x_pos[slot]],
                [node_xy[(slot, 'strong')][1], node_xy[(slot, 'weak')][1]],
                color='#999999', linewidth=1, zorder=1)

        # Draw the final tournament winner
        if slot == final_slot(tree):
            sx, sy, s_team, s_extra, _ = node_xy[(slot, None)]
            if s_team is not None:
                color = slot_color_fn(slot, s_team)
                label = _team_label(s_team, team_names, seeds, False)
                if s_extra:
                    label = f'{label} ({s_extra})'
                
                # Extend a short line for the overall champion
                ax.plot([sx, sx + 1], [sy, sy], color='#999999', linewidth=1, zorder=1)
                ax.text(sx + 0.05, sy + 0.05, label, ha='left', va='bottom',
                        fontsize=7, color=color, fontweight='bold', zorder=2)

    max_round = max(node['round'] for node in tree.values())
    ax.set_xlim(-1.5, max_round + 3)
    ax.set_xticks(range(-1, max_round + 1))
    ax.set_xticklabels(['team'] + [f'round {r}' if r > 0 else 'play-in' for r in range(0, max_round + 1)])
    ax.set_yticks([])
    ax.set_title(title)


def plot_bracket(tree: dict, winners: dict, actual_winners: Optional[dict] = None, team_names: Optional[dict] = None,
                  seeds: Optional[dict] = None, title: Optional[str] = None, ax: Optional[Axes] = None, figsize: Optional[tuple] = None) -> Axes:
    '''draws one simulated bracket, coloring each pick green/red against the actual result if provided

    Parameters
    ----------
    tree : dict
        bracket tree as built by bracket.build_bracket_tree
    winners : dict
        slot -> predicted winning team id, e.g. one simulation from bracket.simulate_bracket
    actual_winners : dict, optional
        slot -> actual winning team id, as returned by bracket.resolve_actual_winners, by default
        None (picks are drawn in a single neutral color with no correctness callout)
    team_names : dict, optional
        team_id -> display name lookup, by default None (falls back to raw team ids)
    seeds : dict, optional
        team_id -> seed lookup, used to prefix first-round entrants, by default None
    title : str, optional
        title to set on the axes, by default None ('Simulated bracket')
    ax : Axes, optional
        axes to draw on, by default None (a new figure is created)
    figsize : tuple, optional
        figure size if a new figure is created, by default None (sized to fit the bracket depth)

    Returns
    -------
    Axes
        the axes the bracket was drawn on
    '''
    n_leaves = sum(1 for node in tree.values()
                   for ref in (node['strong'], node['weak']) if not is_slot_ref(ref))
    if figsize is None:
        figsize = (16, max(6, 0.35 * max(n_leaves, 1)))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    def label_fn(slot, team_leaf):
        if slot is not None:
            pred = winners.get(slot)
            extra = None
            # Fetch and display actual winner if prediction was wrong
            if actual_winners and slot in actual_winners and actual_winners[slot] != pred:
                actual = actual_winners[slot]
                actual_name = _team_label(actual, team_names)
                extra = f"Act: {actual_name}"
            return pred, extra
        return team_leaf, None

    x_pos, node_xy = _walk_layout(tree, label_fn)

    def color_fn(slot, team):
        if actual_winners is None or slot not in actual_winners:
            return '#1f77b4'
        correct = team == actual_winners.get(slot)
        return '#2ca02c' if correct else '#d62728'

    _draw_tree(tree, x_pos, node_xy, team_names, title or 'Simulated bracket', ax, color_fn, seeds)

    if actual_winners is not None:
        ax.plot([], [], color='#2ca02c', label='matches actual result')
        ax.plot([], [], color='#d62728', label='differs from actual result, Predicted winner (Act: Observed winner)')
        ax.legend(loc='lower right', fontsize=8)
    return ax


def plot_slot_win_probabilities(tree: dict, all_sims: list[dict], actual_winners: Optional[dict] = None, team_names: Optional[dict] = None,
                                 seeds: Optional[dict] = None, title: Optional[str] = None, ax: Optional[Axes] = None, figsize: Optional[tuple] = None) -> Axes:
    '''draws a bracket showing each slot's most likely winner across many simulations, with its win probability

    Parameters
    ----------
    tree : dict
        bracket tree as built by bracket.build_bracket_tree
    all_sims : list[dict]
        list of simulated brackets, each slot -> winning team id, as returned by bracket.simulate_bracket
    actual_winners : dict, optional
        slot -> actual winning team id, as returned by bracket.resolve_actual_winners, by default
        None (top picks are drawn in a single neutral color with no correctness callout)
    team_names : dict, optional
        team_id -> display name lookup, by default None (falls back to raw team ids)
    seeds : dict, optional
        team_id -> seed lookup, used to prefix first-round entrants, by default None
    title : str, optional
        title to set on the axes, by default None (defaults to a count of simulations)
    ax : Axes, optional
        axes to draw on, by default None (a new figure is created)
    figsize : tuple, optional
        figure size if a new figure is created, by default None (sized to fit the bracket depth)

    Returns
    -------
    Axes
        the axes the bracket was drawn on
    '''
    n_leaves = sum(1 for node in tree.values()
                   for ref in (node['strong'], node['weak']) if not is_slot_ref(ref))
    if figsize is None:
        figsize = (16, max(6, 0.35 * max(n_leaves, 1)))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    def label_fn(slot, team_leaf):
        if slot is not None:
            dist = slot_winner_distribution(all_sims, slot)
            top_team, top_prob = max(dist.items(), key=lambda kv: kv[1])
            
            extra_str = f'{top_prob:.0%}'
            if actual_winners and slot in actual_winners and actual_winners[slot] != top_team:
                actual = actual_winners[slot]
                actual_name = _team_label(actual, team_names)
                extra_str += f" | Act: {actual_name}"
                
            return top_team, extra_str
        return team_leaf, None

    x_pos, node_xy = _walk_layout(tree, label_fn)

    def color_fn(slot, team):
        if actual_winners is None or slot not in actual_winners:
            return '#1f77b4'
        correct = team == actual_winners.get(slot)
        return '#2ca02c' if correct else '#d62728'

    _draw_tree(tree, x_pos, node_xy, team_names,
               title or f'Slot win probabilities across {len(all_sims)} simulations', ax, color_fn, seeds)

    if actual_winners is not None:
        ax.plot([], [], color='#2ca02c', label='top pick matches actual result')
        ax.plot([], [], color='#d62728', label='top pick differs from actual result')
        ax.legend(loc='lower right', fontsize=8)
    return ax

def plot_upset_calls(tree: dict, hybrid_winners: dict, chalk_winners: dict, actual_winners: Optional[dict] = None,
                      team_names: Optional[dict] = None, seeds: Optional[dict] = None, title: Optional[str] = None,
                      ax: Optional[Axes] = None, figsize: Optional[tuple] = None) -> Axes:
    '''draws a bracket highlighting where the hybrid strategy's picks diverge from chalk (i.e. its upset calls)

    Parameters
    ----------
    tree : dict
        bracket tree as built by bracket.build_bracket_tree
    hybrid_winners : dict
        slot -> predicted winning team id from the hybrid strategy
    chalk_winners : dict
        slot -> predicted winning team id from chalk, used as the comparison baseline
    actual_winners : dict, optional
        slot -> actual winning team id, as returned by bracket.resolve_actual_winners, by default
        None (upset calls are drawn in a single neutral color with no correctness callout)
    team_names : dict, optional
        team_id -> display name lookup, by default None (falls back to raw team ids)
    seeds : dict, optional
        team_id -> seed lookup, used to prefix first-round entrants, by default None
    title : str, optional
        title to set on the axes, by default None ('Hybrid picks: upset calls vs chalk')
    ax : Axes, optional
        axes to draw on, by default None (a new figure is created)
    figsize : tuple, optional
        figure size if a new figure is created, by default None (sized to fit the bracket depth)

    Returns
    -------
    Axes
        the axes the bracket was drawn on
    '''
    n_leaves = sum(1 for node in tree.values()
                   for ref in (node['strong'], node['weak']) if not is_slot_ref(ref))
    if figsize is None:
        figsize = (16, max(6, 0.35 * max(n_leaves, 1)))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    def label_fn(slot, team_leaf):
        if slot is not None:
            pick = hybrid_winners.get(slot)
            extra = None
            if chalk_winners.get(slot) != pick:
                chalk_pick = int(chalk_winners.get(slot)) # type: ignore
                chalk_name = _team_label(chalk_pick, team_names)
                extra = f"upset call, chalk: {chalk_name}"
            return pick, extra
        return team_leaf, None

    x_pos, node_xy = _walk_layout(tree, label_fn)

    def color_fn(slot, team):
        is_upset_call = chalk_winners.get(slot) != team
        if not is_upset_call:
            return '#1f77b4'  # agrees with chalk
        if actual_winners is None or slot not in actual_winners:
            return '#ff7f0e'  # upset call, outcome unknown/not being scored
        return '#2ca02c' if team == actual_winners.get(slot) else '#d62728'  # upset call, right / wrong

    _draw_tree(tree, x_pos, node_xy, team_names,
               title or 'Hybrid picks: upset calls vs chalk', ax, color_fn, seeds)

    ax.plot([], [], color='#1f77b4', label='agrees with chalk')
    if actual_winners is not None:
        ax.plot([], [], color='#2ca02c', label='upset call, correct')
        ax.plot([], [], color='#d62728', label='upset call, incorrect')
    else:
        ax.plot([], [], color='#ff7f0e', label='upset call')
    ax.legend(loc='lower right', fontsize=8)
    return ax

def build_seed_dict(season, seed_df) -> dict:
    if 'seed' not in seed_df.columns:
            from src.data.dataset import parse_seed
            seed_df = seed_df.copy()
            seed_df['seed'] = seed_df['Seed'].apply(parse_seed)
    season_df = seed_df[seed_df['Season'] == season]
    return pd.Series(season_df.seed.values, index=season_df.TeamID).to_dict()