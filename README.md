# ATP Tennis Match Prediction

Predicting the winner of ATP tennis matches using only information available
**before** the match: rankings, points, recent form, surface history,
head-to-head records, player biography and bookmaker odds.

University group project (Artificial Intelligence course), built in Python with
pandas, scikit-learn and CatBoost on ~28,000 ATP matches from 2015 to 2026.

## Highlights

- **Leakage-free features**: every historical feature (win rate, last-five form,
  surface record, rest days, head-to-head) is computed from strictly earlier
  dates only. Post-match score columns are dropped before feature engineering.
- **Symmetric dataset**: each match appears twice, once from each player's
  point of view, with complementary labels. This removes the bias of a
  "winner always in column A" layout and gives a perfectly balanced target.
- **Chronological evaluation**: the newest 15% of distinct dates is held out
  and never used for tuning. Cross-validation folds are built on dates, so
  both orientations of a match always stay in the same fold.
- **Data enrichment**: Tennis-Data match records are joined with Jeff
  Sackmann's ATP player database (height, handedness, age) through fuzzy name
  matching, with every accepted and rejected match logged for review.
- **Honest benchmarking** against a naive baseline and against the
  bookmakers' implied probabilities on the same rows.

## Results (holdout: Nov 2024 – Jul 2026)

| Model | Accuracy | ROC AUC |
| --- | ---: | ---: |
| Naive baseline | 0.500 | 0.500 |
| Decision Tree (Gini, grid search) | 0.686 | — |
| CatBoost | 0.686 | 0.751 |
| Bookmaker odds (same rows) | 0.688 | 0.751 |

Both models reach roughly the accuracy of the betting market, which is
already a very strong predictor. The models do not beat the market, and these
results should not be read as a betting strategy. Full metrics (log loss,
Brier score, match-level accuracy) are in
[docs/TECHNICAL.md](docs/TECHNICAL.md).

## My contribution

I worked on the data pipeline and on the interpretable model:

- **Feature engineering** (`scripts/data_cleaner/build_features.py`):
  pre-match player histories, head-to-head, surface and form features,
  the symmetric target construction, and a Random Forest + RFECV baseline for
  feature selection.
- **Sackmann enrichment** (`scripts/data_cleaner/enrich_with_sackmann.py`):
  matching Tennis-Data player names to Sackmann IDs and adding biography data.
- **Decision Tree** (`scripts/models/train_decision_tree.py` and
  `docs/decision_tree_gini_training_analysis_it_3.ipynb`): Gini tree with
  grid search over depth, leaf size and cost-complexity pruning, date-based
  CV folds, a comparison with and without market-odds features, feature
  importance and error analysis.

## Team

- **Alberto Campagnolo** ([@K4mp47](https://github.com/K4mp47)): data
  preparation and normalization, CatBoost model and analysis notebooks.
- **Gianmarco Gandini**: feature engineering, data enrichment, Decision Tree.

Original repository: [K4mp47/tennis_prediction](https://github.com/K4mp47/tennis_prediction).

## Repository structure

```text
scripts/
  data_downloader/   download and merge yearly Tennis-Data workbooks
  data_cleaner/      normalization, Sackmann enrichment, feature engineering
  models/            Decision Tree and CatBoost training, prediction
docs/
  TECHNICAL.md       full pipeline, assumptions and metrics
  *.ipynb            analysis notebooks (Decision Tree, CatBoost)
```

## Running it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Data files are not
included in the repository; see [docs/data_sources.md](docs/data_sources.md)
for where to get them and [docs/TECHNICAL.md](docs/TECHNICAL.md) for the full
step-by-step pipeline.

```bash
uv sync --all-groups
uv run python scripts/models/train_decision_tree.py
```

## Data and licensing

- Match data: [Tennis-Data](http://tennis-data.co.uk/).
- Player biography: Jeff Sackmann's
  [tennis_atp](https://github.com/JeffSackmann/tennis_atp) dataset, licensed
  CC BY-NC-SA 4.0 (attribution required, non-commercial use only).
