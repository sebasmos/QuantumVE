import argparse
import pandas as pd
import os

def merge_results(results_dir):
    all_files = []
    
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            for seed_folder in os.listdir(folder_path):
                seed_folder_path = os.path.join(folder_path, seed_folder)
                if os.path.isdir(seed_folder_path):
                    for file in os.listdir(seed_folder_path):
                        if file == 'average_metrics_mlp.csv':
                            file_path = os.path.join(seed_folder_path, file)
                            df = pd.read_csv(file_path)
                            df['Folder'] = folder
                            df['Seed'] = seed_folder
                            all_files.append(df)
    
    all_data = pd.concat(all_files, ignore_index=True)
    
    merged_file_path = os.path.join(results_dir, 'merged_results.csv')
    all_data.to_csv(merged_file_path, index=False)
    print(f'Merged results saved to {merged_file_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge results from multiple folders and seeds.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing the results.')
    args = parser.parse_args()
    
    merge_results(args.results_dir)