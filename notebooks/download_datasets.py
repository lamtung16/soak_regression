import os
import pandas as pd
import requests

# ---- CONFIG ----
INPUT_CSV = "results.csv"
OUTPUT_DIR = "test_data"
HF_API_URL = "https://huggingface.co/api/datasets"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- LOAD ----
df = pd.read_csv(INPUT_CSV)

# ---- FILTER ----
mask = df["cols"].str.contains(r"\b(sex|gender)\b", case=False, na=False)
datasets = df.loc[mask, "dataset"].dropna().unique()

print(f"Found {len(datasets)} matching datasets")

# ---- HELPERS ----
def list_files(dataset_name):
    """Get list of files in Hugging Face dataset repo"""
    url = f"{HF_API_URL}/{dataset_name}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to fetch metadata: {dataset_name}")
        return []

    data = r.json()

    files = []
    for sibling in data.get("siblings", []):
        fname = sibling.get("rfilename", "")
        if fname.endswith(".csv"):
            files.append(fname)

    return files


def download_file(dataset_name, file_path):
    """Download a single CSV file"""
    url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{file_path}"
    
    local_dir = os.path.join(OUTPUT_DIR, dataset_name.replace("/", "_"))
    os.makedirs(local_dir, exist_ok=True)

    local_path = os.path.join(local_dir, os.path.basename(file_path))

    print(f"Downloading {url}")

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"Failed: {url} -> {e}")


# ---- MAIN LOOP ----
for ds in datasets:
    print(f"\nProcessing: {ds}")
    files = list_files(ds)

    if not files:
        print("No CSV files found")
        continue

    for f in files:
        download_file(ds, f)

print("\nDone.")