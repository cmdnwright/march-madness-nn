'''
orchestrates the loso bracket backtest simulations, simulating n_sims brackets per held out season.
builds bracket per season using slots and recursively walks from initial teams to championship.

score is based on espn scoring * 10 for legibility

saves results of form season, model, mean_score, std_score, scores to data/processed/loso_results.csv

included models: chalk, random, lr, rf, nn_baseline, nn_focal, hybrid_upset (chalk w/ nn predict upsets)

does not require any pretrained nns, only baseline and upset data pipelines to have run 
(team_stats, splits for base and upset data experiments)

run using python -m scripts.run_loso_backtest.py
'''


import pandas as pd
from src.utils.config import load_config
from src.data.ingest import load_raw_data
from src.simulation.bracket import load_slots
from src.simulation.loso import run_loso_backtest, summarize_loso_results

N_SIMS = 1000
RANDOM_SEED = 42
FRESH = False
CONFIG_PATHS = [
    'configs/baseline.yaml',
    'configs/focal.yaml',
]

UPSET_CONFIG_PATH = 'configs/upset.yaml'
UPSET_THRESHOLD = 0.5
OUTPUT_PATH = 'data/processed/loso_results.csv'


def main():
    configs = [load_config(path) for path in CONFIG_PATHS]
    base_config = configs[0]  # shared data/feature settings across all NN variants
    upset_config = load_config(UPSET_CONFIG_PATH) if UPSET_CONFIG_PATH else None

    raw = load_raw_data(base_config['data']['raw_dir'])
    tourney_results = raw['tourney_results']
    seeds = raw['seeds']
    slots_df = load_slots(base_config['data']['raw_dir'])

    experiment_name = base_config['data']['experiment_name']
    team_stats = pd.read_csv(f'{base_config["data"]["processed_dir"]}/team_stats_{experiment_name}.csv')

    results = run_loso_backtest(
        configs, team_stats, seeds, tourney_results, slots_df,
        output_path=OUTPUT_PATH,
        n_sims=N_SIMS,
        random_seed=RANDOM_SEED,
        resume=not FRESH,
        upset_config=upset_config,
        upset_threshold=UPSET_THRESHOLD,
    )

    print(summarize_loso_results(results))


if __name__ == '__main__':
    main()