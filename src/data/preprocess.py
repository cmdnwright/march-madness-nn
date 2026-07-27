import pandas as pd

def build_team_stats(config, regular_season):
    wins_df = regular_season[['Season', 'WTeamID', 'WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].copy()
    wins_df.columns = ['season', 'team_id', 'points_scored', 'points_allowed', 'shots_made', 'shots_attempted', 'threes_made', 'threes_attempted', 'free_throws_made', 'free_throws_attempted', 'offensive_rebounds', 'defensive_rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'fouls']
    wins_df['wins'] = 1

    loss_df = regular_season[['Season', 'LTeamID', 'LScore', 'WScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].copy()
    loss_df.columns = ['season', 'team_id', 'points_scored', 'points_allowed', 'shots_made', 'shots_attempted', 'threes_made', 'threes_attempted', 'free_throws_made', 'free_throws_attempted', 'offensive_rebounds', 'defensive_rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'fouls']
    loss_df['wins'] = 0

    games = pd.concat([wins_df, loss_df])
    games['games'] = 1

    team_stats = games.groupby(['season', 'team_id']).agg(
        wins = ('wins', 'sum'),
        games_played = ('wins', 'count'),
        points_scored = ('points_scored', 'mean'),
        points_allowed = ('points_allowed', 'mean'),
        shots_made = ('shots_made', 'mean'),
        shots_attempted = ('shots_attempted', 'mean'),
        threes_made = ('threes_made', 'mean'),
        threes_attempted = ('threes_attempted', 'mean'),
        free_throws_made = ('free_throws_made', 'mean'),
        free_throws_attempted = ('free_throws_attempted', 'mean'),
        offensive_rebounds = ('offensive_rebounds', 'mean'),
        defensive_rebounds = ('defensive_rebounds', 'mean'),
        assists = ('assists', 'mean'),
        turnovers = ('turnovers', 'mean'),
        steals = ('steals', 'mean'),
        blocks = ('blocks', 'mean'),
        fouls = ('fouls', 'mean')
    ).reset_index()

    team_stats['win_pct'] = team_stats['wins'] / team_stats['games_played']
    team_stats['fg_pct'] = team_stats['shots_made'] / team_stats['shots_attempted']
    team_stats['three_pct'] = team_stats['threes_made'] / team_stats['threes_attempted']
    team_stats['free_throw_pct'] = team_stats['free_throws_made'] / team_stats['free_throws_attempted']

    team_stats.to_csv(f'{config["data"]["processed_dir"]}/team_stats.csv', index=False)