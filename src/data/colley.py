import numpy as np
import pandas as pd


def compute_colley_ratings(regular_season: pd.DataFrame) -> pd.DataFrame:
    '''computes colley ratings per team per season from regular season results

    builds the colley linear system per season (C r = b) and solves it directly,
    since colley ratings are only meaningfully compared within a single season

    Parameters
    ----------
    regular_season : pd.DataFrame
        regular season results df formatted by season and WTeamID LTeamID WScore LScore

    Returns
    -------
    pd.DataFrame
        colley ratings df of the form season, team_id, colley

    Raises
    ------
    ValueError
        the regular season results contain a tied game (WScore == LScore), which the
        colley system has no defined behavior for
    '''
    tied_games = regular_season[regular_season['WScore'] == regular_season['LScore']]
    if len(tied_games) > 0:
        raise ValueError(
            f"Found {len(tied_games)} games {tied_games[['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore']]}"
        )

    all_ratings = []

    for season, season_games in regular_season.groupby('Season'):
        teams = pd.unique(season_games[['WTeamID', 'LTeamID']].values.ravel())
        teams = np.sort(teams)
        n = len(teams)
        team_to_idx = {team_id: i for i, team_id in enumerate(teams)}

        C = np.zeros((n, n), dtype=np.float64)
        wins = np.zeros(n, dtype=np.float64)
        losses = np.zeros(n, dtype=np.float64)

        # build the colley matrix and win/loss counts game by game
        for _, game in season_games.iterrows():
            i = team_to_idx[game['WTeamID']]
            j = team_to_idx[game['LTeamID']]

            C[i, i] += 1
            C[j, j] += 1
            C[i, j] -= 1
            C[j, i] -= 1

            wins[i] += 1
            losses[j] += 1

        C += np.diag(np.full(n, 2.0))
        b = 1.0 + (wins - losses) / 2.0

        # C is strictly diagonally dominant by construction, so this solve is always well-posed, no regularization needed.
        r = np.linalg.solve(C, b)

        season_ratings = pd.DataFrame({
            'season': season,
            'team_id': teams,
            'colley': r
        })
        all_ratings.append(season_ratings)

    return pd.concat(all_ratings, ignore_index=True)