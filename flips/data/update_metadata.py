import csv
import argparse
from pathlib import Path
from tqdm import tqdm

def find_all_files(target_dir):
    allowed_extensions = ['.npz', '.pkl']
    file_map = {}
    target_path = Path(target_dir)
    for ext in allowed_extensions:
        all_files = list(target_path.rglob(f'*{ext}'))
        for file_path in all_files:
            file_map[file_path.name] = file_path
    if not file_map:
        raise ValueError(f"No files with extensions {allowed_extensions} found in {target_dir}")
    return file_map

def update_csv_paths(csv_file, target_dir=None):
    csv_path = Path(csv_file)
    assert csv_path.exists(), f'File {csv_file} does not exist.'
    if target_dir is None:
        target_dir = csv_path.parent
    base_path = Path(target_dir)
    updated_rows = []

    file_map = find_all_files(base_path)

    with open(csv_file, mode='r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        updated_rows.append(header)
        rows = list(reader)

    for row in tqdm(rows, desc='Updating CSV paths'):
        for i, item in enumerate(row):
            suffix = Path(item).suffix
            if suffix in ['.npz', '.pkl']:
                filename = Path(item).name
                if filename in file_map:
                    row[i] = str(file_map[filename])
                else:
                    raise FileNotFoundError(f"{filename} not found in {base_path}")
        updated_rows.append(row)

    with open(csv_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(updated_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update .npz file paths in a CSV file')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file.')
    parser.add_argument('--directory', type=str, default=None, help='Directory to search for .npz or .pkl files (default: csv_file parent directory).')
    args = parser.parse_args()

    update_csv_paths(args.csv_file, args.directory)