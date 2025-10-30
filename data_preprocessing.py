import pandas as pd
import os

def preprocess_all_csvs(data_folder="data"):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    for file_name in csv_files:
        file_path = os.path.join(data_folder, file_name)
        print(f"Processing {file_name}...")
        
        # Read CSV
        df = pd.read_csv(file_path)
        original_df = df.copy()
        
        # Example preprocessing:
        df.ffill(inplace=True)                  # Forward fill missing values
        df.drop_duplicates(inplace=True)        # Remove duplicates
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()       # Strip whitespace
        
        # Only overwrite if there are changes
        if df.equals(original_df):
            print(f"  No changes needed.")
        else:
            df.to_csv(file_path, index=False)
            print(f"  Problems found and fixed. Updated CSV saved.")

# Run the preprocessing on all CSVs in the folder
preprocess_all_csvs("data")