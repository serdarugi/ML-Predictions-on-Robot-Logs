# ML on Robots: Queue Time and Execution Time Prediction

This project builds a machine learning workflow for predicting robot mission timing in an automated operation environment.

The main goal is to estimate:

- **Queue time**: time between mission creation and mission start
- **Execution time**: time between mission start and mission completion

The workflow uses mission metadata, robot information, and time-based features to train regression models.

## Project Structure

```text
.
├── main.ipynb
├── test_synthetic_15000.csv
├── generate_synthetic_dataset.py
└── sentetic.py
```

## Dataset

The main dataset is:

```text
test_synthetic_15000.csv
```

It contains 15,000 robot mission records with the following raw columns:

- `Sequence`
- `Mission Name`
- `Robot Name`
- `Map Id`
- `Robot Id`
- `Source`
- `State`
- `State Outcome`
- `Created At`
- `Started At`
- `Completed At`

The dataset is semicolon-separated (`;`) and encoded as UTF-8.

## Synthetic Data Generation

`generate_synthetic_dataset.py` expands the original mission data into a 15,000-row dataset.

The script preserves the original schema and generates synthetic records by sampling mission patterns, queue durations, execution durations, timestamp gaps, and sequence increments from the original data distribution.

Run:

```bash
python generate_synthetic_dataset.py
```

Expected output:

```text
test_synthetic_15000.csv
```

## Notebook Workflow

The main analysis and modeling workflow is in:

```text
main.ipynb
```

The notebook follows this general process:

1. Load the synthetic CSV dataset
2. Convert timestamp columns to datetime
3. Create target variables:
   - `queue_time_seconds`
   - `execution_time_seconds`
4. Extract mission grouping features from `Mission Name`
5. Remove unwanted mission types such as `YarımPalet` and `Other`
6. Extract robot number from `Robot Name`
7. Create time-based features from `Created At`
8. Train regression models for queue time and execution time
9. Evaluate model performance on test data

## Target Variables

The project predicts two target variables:

```text
queue_time_seconds = Started At - Created At
execution_time_seconds = Completed At - Started At
```

These targets are computed using full datetime values, not only the time portion. This avoids incorrect negative durations when missions pass midnight.

## Feature Engineering

The notebook creates mission-level and time-level features.

Mission grouping examples:

```text
[Z1L] DoluPalet --> Streç       -> Z1L Streç
[Z2R] DoluPalet --> Kolon       -> Z2R Kolon
[Z3L] DoluPalet --> ÇelikPalet  -> Z3L ÇelikPalet
```

Time-based features are extracted from `Created At`:

- `created_at_hour`
- `created_at_minute`
- `created_at_month`
- `created_at_dayofweek`

`Started At` and `Completed At` should not be used as model input features because they directly define the target variables and would cause data leakage.

## Modeling Approach

The recommended approach is to train two separate regression models:

- Queue time prediction model
- Execution time prediction model

Example input features:

- `Mission Group`
- `Robot#`
- `created_at_hour`
- `created_at_minute`
- `created_at_month`
- `created_at_dayofweek`

Example targets:

- `queue_time_seconds`
- `execution_time_seconds`

Categorical features should be encoded with `OneHotEncoder`. Numeric features can either be passed through directly or scaled depending on the model type.

For tree-based models such as `RandomForestRegressor`, scaling is usually not required.

## Evaluation

Recommended evaluation metrics:

- **MAE**: average absolute prediction error
- **RMSE**: penalizes larger errors more strongly
- **R2 Score**: explains how much variance the model captures

MAE is especially useful because it can be interpreted directly in seconds or converted to minutes.

Example:

```text
MAE = 120 seconds -> average error is 2 minutes
```

## Data Leakage Notes

Do not use the following columns as model input features:

- `Started At`
- `Completed At`
- `queue_time_seconds`
- `execution_time_seconds`

`Started At` and `Completed At` are only used to calculate the target variables. They are not known at prediction time.

## Requirements

Recommended Python packages:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## How to Run

1. Clone the repository.
2. Install the required Python packages.
3. Open `main.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the notebook cells from top to bottom.

Optional: regenerate the synthetic dataset first:

```bash
python generate_synthetic_dataset.py
```

Then run:

```bash
jupyter notebook main.ipynb
```

## Current Status

The project currently includes:

- Synthetic dataset generation
- Preprocessing and feature engineering workflow
- Queue time and execution time target creation
- Initial modeling setup for regression-based prediction

Future improvements may include hyperparameter tuning, model comparison, feature importance analysis, and saving trained models for inference.
