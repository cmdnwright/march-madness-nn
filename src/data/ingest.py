import pandas as pd
from pathlib import Path

def load_raw_data(data_dir: str) -> dict[str, pd.DataFrame]:
    '''loads the raw data frames from the kaggle data

    Parameters
    ----------
    data_dir : str
        directory of the raw data relative to the root

    Returns
    -------
    dict[str, pd.DataFrame]
        dictionary of strings to dataframes

        * 'tourney_results' : the tournament results for all seasons
            includes Season, WTeamID, WScore, LTeamID, LScore

        * 'season_results' : regular season results across all seasons by game
            includes WTeamID, LTeamID, and all game metrics broken down by W and L team
        
        * 'seeds' : tournament seeds by season
            includes seed and corresponding team id
    '''
    return {
        "tourney_results": pd.read_csv(Path(data_dir) / "MNCAATourneyCompactResults.csv"),
        "season_results": pd.read_csv(Path(data_dir) / "MRegularSeasonDetailedResults.csv"),
        'seeds': pd.read_csv(Path(data_dir) / 'MNCAATourneySeeds.csv')
    }
