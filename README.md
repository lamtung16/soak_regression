# SOAK: Same/Other/All K-fold Cross-Validation
SOAK is designed to estimate the **similarity of patterns** found across different subsets of a dataset. It extends traditional K-fold cross-validation with "Same," "Other," and "All" splitting strategies to provide a robust measure of pattern similarity.

## Pseudocode
```
FOR each subset of the dataset:
    Split the subset into folds (e.g, 5)

    FOR each test fold:
        Define train sets:
            - "same": data from the current subset (excluding the test fold)
            - "other": data from other subsets
            - "all": combination of "same" and "other"

        FOR each train set in ("same", "other", "all"):
            FOR each model (e.g, featureless, linear, tree, ...):
                The model is trained on the train set and evaluated on the test fold
                Record subset, train set category, fold, model, evaluation metrics (e.g, MSE, MAE)
```

## Usage

### 0. Folder Structure
- **`data`**: Contains all datasets in CSV format. Feature columns has to start with 'X_'
- **`results`**: Contains CSV files of computed errors for each dataset.  
- **`figures`**: Contains figures of errors for each dataset.  
- **`notebooks`**: Jupyter notebooks for testing.


### 1. Generate Dataset Parameters
Create params.csv with columns: `dataset`,`subset_col`,`target_col`

### 2. Generate Results
Use `results_generator.py` to generate results and figures for a specific row in `params.csv`:
```bash
python results_generator.py i
```

## Figures
![alt text](figures/synthetic/subset/sum.png)

## TODO
- Find more regression datasets:
   - source: uci repo or openml
   - having different categories (such as gender)
   - prioritize datasets having many citations
