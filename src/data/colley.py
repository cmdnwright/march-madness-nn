import numpy as np
import pandas as pd


def compute_colley_ratings(regular_season: pd.DataFrame) -> pd.DataFrame:
    tied_games = regular_season[regular_season['WScore'] == regular_season['LScore']]
    if len(tied_games) > 0:
        raise ValueError(
            f'Found {len(tied_games)} game(s) with WScore == LScore, which '
            f'violates the win/loss assumption required by the Colley system. '
            f'Inspect these rows before proceeding — do not silently drop or '
            f'split them, since Colley has no defined behavior for a tie:\n'
            f"{tied_games[['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore']]}"
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

        # C is strictly diagonally dominant by construction, so this solve
        # is always well-posed — no regularization needed.
        r = np.linalg.solve(C, b)

        season_ratings = pd.DataFrame({
            'season': season,
            'team_id': teams,
            'colley': r
        })
        all_ratings.append(season_ratings)

    return pd.concat(all_ratings, ignore_index=True)