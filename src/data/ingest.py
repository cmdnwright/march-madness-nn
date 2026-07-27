import pandas as pd
from pathlib import Path

def load_raw_data(data_dir: str) -> dict[str, pd.DataFrame]:
    return {
        "tourney_results": pd.read_csv(Path(data_dir) / "MNCAATourneyCompactResults.csv"),
        "season_results": pd.read_csv(Path(data_dir) / "MRegularSeasonDetailedResults.csv"),
        'seeds': pd.read_csv(Path(data_dir) / 'MNCAATourneySeeds.csv')
    }
