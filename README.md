# Predicting Smartphone Addiction

This project builds binary classification models for a Kaggle-style smartphone addiction prediction competition. The goal is to estimate the probability that an individual is addicted (`addicted_label`) from demographic, screen-use, work/study, sleep, and notification features.

## Dataset

The project includes the following data files:

- `train.csv`: 691,369 labelled rows
- `test.csv`: 296,302 rows for prediction
- `sample_submission.csv`: expected submission format

The target column is `addicted_label`. The positive-class prevalence in the training data is approximately 70.94%.

### Input Features

- `age`
- `daily_screen_time_hours`
- `social_media_hours`
- `gaming_hours`
- `work_study_hours`
- `sleep_hours`
- `notifications_per_day`
- `app_opens_per_day`
- `weekend_screen_time`
- `gender`
- `stress_level`
- `academic_work_impact`
- `id`

## Project Files

- `train_ensemble.py`: trains LightGBM, CatBoost, and XGBoost with 5-fold stratified cross-validation, blends out-of-fold predictions, searches an optimal classification threshold, and writes submission files.
- `train_catboost.py`: trains and validates a standalone CatBoost model and writes a probability-based submission.
- `code.ipynb`: exploratory notebook and model-development workflow.
- `submission.csv`: ensemble probability predictions.
- `final_submission.csv`: binary ensemble predictions using the selected threshold.
- `cat_sub.csv`: standalone CatBoost probability predictions.
- `model_metrics.npz`: saved out-of-fold predictions, test predictions, and ensemble metrics.

## Engineered Features

The scripts create behavioural features such as:

- Total screen activity
- Weekend screen-time difference
- Screen/sleep and screen/work ratios
- Social and gaming activity ratios
- Notifications and app opens per screen hour
- Missing-value count
- Age and screen-time interaction

## Setup

Create a Python environment and install the required libraries:

```bash
python -m pip install numpy pandas scikit-learn catboost lightgbm xgboost matplotlib seaborn
```

Run the scripts from the project directory:

```bash
python train_catboost.py
python train_ensemble.py
```

Both scripts read `train.csv` and `test.csv` from the project directory. The ensemble script must be run from the project root because it uses relative output paths.

## Ensemble Method

`train_ensemble.py` uses:

- 5-fold stratified cross-validation
- LightGBM
- CatBoost
- XGBoost
- 40% LightGBM / 35% CatBoost / 25% XGBoost probability blend
- Threshold optimization over 0.35–0.65

It reports accuracy and ROC-AUC for each model and the ensemble, then creates:

- `submission.csv`: probability predictions in Kaggle submission format
- `final_submission.csv`: 0/1 predictions using the optimized threshold

Submission columns are:

```text
id,addicted_label
```

## Standalone CatBoost

`train_catboost.py` performs a 90/10 stratified validation split, reports validation log loss and ROC-AUC, refits on all labelled rows using the early-stopping iteration count, and writes `cat_sub.csv`.
