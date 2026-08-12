# A Comparative Study of Classical and Neural Approaches for Tournament Predictions

A rigorous, statistically grounded evaluation of machine learning methods for predicting NCAA Men's Basketball Tournament outcomes. Predictions are made at the individual game level and the full-bracket level using 20 seasons of regular season and tournament data. We first test various feature sets at the individual level before evaluating models, including our best upset-prediction-hybrid model, at the tournament level.

### Key Findings
No model in this study achieves a statistically significant improvement over the seed heuristic (McNemar's test, Holm-Bonferroni corrected) at the individual game level, and no bracket-simulation strategy statistically outperforms simply picking the higher seed every round (Wilcoxon signed rank test, Holm-Bonferroni corrected), with the exception a hybrid seed/upset model that matches chalk's expected score while reducing its variance.

## Motivation

Predicting tournament outcomes is a compelling problem because it is an obvious application of standard machine learning methods to a notoriously challenging question year over year. Single elimination tournaments produce ~1500 total games across 20 seasons, meaning any given season is a small, noisy sample. Upsets are common enough that the 'obvious' prediction (better seed wins or 'chalk') is wrong ~20% of the time depending on the year. Despite this, many strategies perform worse than the chalk predictions year over year because the seed is a well engineered feature that already contains a significant amount of predictive information. This project seeks to rigorously compare strategies for predicting tournament outcomes through tests of feature validity under leakage constraints, model comparison via hypothesis testing rather than point estimates, and evaluations of a strategy's outcome distribution, not just its mean.

## Approaches compared

### Game-Level Win Prediction 
Is Team A more likely to win than Team B? Trained on 2003-2023 seasons, val on 2024 season, test on 2025 season results.

**Model Metrics**
| Model | Accuracy | AUC | Log Loss |
|---|---|---|---|
| Logistic Regression | 0.731 | 0.873 | 0.500 |
| Random Forest (100 trees) | 0.761 | 0.813 | 0.550|
| SVM | 0.746 | 0.805 | 0.564 |
| Neural Network | 0.724 | 0.830 | 0.503 |
| Naive Seed | 0.784 | - | - |

**McNemar's Test Standard p-values**
| Model | LR | RF | SVM |
| :--- | :--- | :--- | :--- |
| **RF** | 0.423950 | - | - |
| **SVM** | 0.790527 | 0.790527 | -|
| **NN** | 1.000000 | 0.332306 | 0.607239 |

**McNemar's Test Holm-Corrected p-values**
| Model | LR | RF | SVM |
| :--- | :--- | :--- | :--- |
| **RF** | 1.000000 | - | - | 
| **SVM** | 1.000000 | 1.000000 | - | 
| **NN** | 1.000000 | 1.000000 | 1.000000 | 

*Table reflects baseline feature set performance. See [`game_simulations.ipynb`](game_simulations.ipynb) for all tested feature sets and [`METHODS.md`](METHODS.md) for feature engineering approach*

### Bracket-Level Simulation 
Leave-one-season-out (LOSO) cross-validation, full 67-game bracket per season, scored on standard ESPN-style points, simulated 1000 brackets per season from 2003-2023.  

| Strategy | Mean season score (out of 1920)| vs chalk p-value | STD | Win rate vs chalk
|---|---|---|---|---|
| Chalk | 862.632 | - | 296.364 | - |
| Hybrid Upset Model | 777.895 | .229 | 209.379 | 0.526 |
| Logistic Regression | 631.951 | ~ 0 | 97.284 | 0.105 |
| Random | 311.226 | ~ 0 | 3.424 | 0.0

*p-values are listed for a Wilcoxon signed rank test after Holm-Bonferroni correction to account increased false positive rate after repeated comparisons*

For the full mathematical derivation and reasoning of all significant feature ideas, model choices, and statistical tests see [**METHODs.md**](METHODs.md).

## Results

- **The seed is a well engineered feature already.** In inference, no baseline model outperforms seeds on single game predictions. No win predicting consistently model outperforms chalk selection in full bracket simulations. This surfaces the approach to use the seed for its predictive power and instead build a strategy around inferring when the seed will be upset. Further, Colley ratings correlate with seed at $r \approx −0.94$ and add no measurable predictive power once seed is in the model. A case study in distinguishing 'predictive of outcome' from 'adds information beyond what's already captured.'

- **Momentum features (trailing 5-season z-scored performance) are not predictive.** AUC hovers near 0.50 despite a plausible sports narrative ('teams get hot').

- **Era normalization is redundant with raw stat differences.** At this data scale the 20-season window doesn't show enough era drift for z-scoring to add value over the raw stat diffs, verified via coefficient-of-variation analysis across seasons. All potential feature screening is reported in [`feature_engineering.ipynb`](feature_engineering.ipynb)

- **No pairwise model comparison at the game level clears statistical significance.**  Point estimates may vary by several accuracy points but McNemar's test fails to confirm statistically significant differences between models, especially After Holm-Bonferroni correction. Bar chart differences on small test sets are well within noise.

- **Predicting upsets directly rather than predicting wins surfaces the only model that's competitive with chalk.** On bracket simulations, predicting upsets to overrule seed differences has insignificant results to show statistically different performance compared to chalk. More importantly, the upset model does so with materially lower score variance. This is a risk/reward tradeoff that a pure accuracy or log-loss metric would miss entirely but is important when considering a strategy.

## Limitations and future work
- Baseline diff based features showed promise in exploratory data analysis, but is a very uncommon feature choice in successful solutions to the Kaggle Competition. Engineering features based on metrics like efficiency while retaining seed information could cut out noise created by less informative features like baseline stat diffs. Further feature engineering analysis should be done to compare predictive power specifically in the context of predicting winners vs predicting upsets.

- All models, specifically win prediction models are explicitly trained to predict single game outcomes. A key challenge of preditcting brackets is understand how current round picks influence future matchups, and a model trained on bracket score rather than BCE for single games could potentially learn how to hedge results or better understand follow-up game trends

- Using a single test season for the single game evaluation leaves a relatively small sample size for statistical evaluation, meaning there is a high likilood our statistical results are under powered rather than genuinely null. Single game simulations would benefit from the same loso cross validation used in bracket simulations

- The hybrid upset model's threshold was selected based on historical upset rate to prevent data leakage. Cross validation hyperparameter tuning could yield better results

- With only 20 tournament seasons of data, statistical power for detecting real but modest effects is inherently limited. Features could be engineered to use more historical data without basing labels on tournament results

## Repository structure

```
march-madness-nn/
|-- configs/ # all model configs
|-- data/
|   |-- raw/ # raw kaggle data
|   |-- processed/ # aggregated stats
|   |-- splits/ # features split train/test/val
|-- models/
|   |-- checkpoints/ # checkpoint ever 10 epochs
|   |-- final/ # lowest val loss
|-- scripts/ # data and training scripts
|-- src/
|   |-- data/ # data modules
|   |-- model/ # model arch and inference
|   |-- simulation/ # bracket and loso sim modules
|   |-- training/ # training loop, loss, eval
|   |-- utils/ # config utils
|-- tests/ # data integrity tests
|-- bracket_simulations.ipynb 
|-- game_simulations.ipynb
|-- feature_engineering.ipynb
|-- exploratory_data_analysis.ipynb
```
All scripts and modules are fully documented. See source files for implementation decisions.

## Reproducing this project

```bash
# 1. clone and install
git clone <repo-url>
pip install -r requirements.txt

# 2. place Kaggle data in data/raw/

# 3. run data script for all winner prediction feature sets
python -m scripts.build_dataset --config configs/baseline.yaml
python -m scripts.build_dataset --config configs/colley.yaml
python -m scripts.build_dataset --config configs/season.yaml

# 4. run data script for upset prediction feature set
python -m scripts.build_upset_dataset --config configs/upset.yaml

# 5. train all desired models
python -m scripts.run_training --config configs/baseline.yaml
python -m scripts.run_training --config configs/colley.yaml
python -m scripts.run_training --config configs/season.yaml
python -m scripts.run_training --config configs/focal.yaml
python -m scripts.run_training --config configs/upset.yaml

# 6. run all notebooks
```

## Data

[Kaggle March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) 
- MNCAATourneyCompactResults.csv
- MNCAATourneySeeds.csv
- MNCAATourneySlots.csv
- MRegularSeasonDetailedResults.csv
- MTeams.csv