import os
import csv

# Path to the folder to scan
folder_path = r"data"  # change this to your folder path

# Get a list of all CSV files in the folder
csv_files = [
    f for f in os.listdir(folder_path)
    if f.lower().endswith('.csv') and os.path.isfile(os.path.join(folder_path, f))
]

# Path to save the output CSV
output_csv_path = r"params.csv"  # change this if needed

# Write CSV file names to the output CSV with a header
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['dataset'])  # header
    for csv_file in csv_files:
        writer.writerow([csv_file[:-4]])  # remove '.csv' extension

print(f"Saved {len(csv_files)} CSV file names to {output_csv_path}")