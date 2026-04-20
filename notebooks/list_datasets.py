from huggingface_hub import list_datasets
from datasets import load_dataset, logging
import pandas as pd

logging.set_verbosity_error()

def has_sex_or_gender(cols):
    cols_lower = [c.lower() for c in cols]
    return any(any(k in c for k in ["sex", "gender"]) for c in cols_lower)

rows = []
datasets = list_datasets(
    filter=[
        "modality:tabular",
        "format:csv",
        "task_categories:tabular-regression"
    ]
)

for ds in datasets:
    name = ds.id
    print(f"Processing: {name}")

    try:
        dset = load_dataset(name, split="train[:1]")
        df = dset.to_pandas()
        rows.append({
            "dataset": name,
            "n_cols": df.shape[1],
            "has_sex_or_gender_column": has_sex_or_gender(df.columns),
            "cols": list(df.columns)
        })

    except Exception as e:
        print(f"Skipping {name}: {e}")

result_df = pd.DataFrame(rows)
result_df.to_csv("results.csv")