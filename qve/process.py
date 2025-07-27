import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import numpy as np


# EDITED: pre-computed data
def data_prepare_cv(n_dim, sample_train, sample_test):
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)
    pca = PCA(n_components=n_dim, svd_solver="auto").fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler(feature_range=(-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)
    return sample_train, sample_test

def data_prepare(n_dim, sample_train, sample_test, nb1, nb2):
    """
    Scales data, applies PCA, and then re-scales using MinMaxScaler.
    nb1 and nb2 control the number of samples returned.
    """
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)
    pca = PCA(n_components=n_dim, svd_solver="auto").fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)
    # Fit MinMaxScaler on the combined data
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler(feature_range=(-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)[:nb1]
    sample_test = minmax_scale.transform(sample_test)[:nb2]
    return sample_train, sample_test

def process_folds(n_folds, input_csv, output_csv):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Filter by number of folds
    filtered_df = df[df['Fold'] <= n_folds]

    # Group by 'n_dim' and calculate mean and std
    mean_metrics = filtered_df.groupby('n_dim').mean(numeric_only=True)
    std_metrics = filtered_df.groupby('n_dim').std(numeric_only=True)

    # Create mean/std results DataFrame
    mean_results = pd.DataFrame({
        'n_dim': mean_metrics.index,
        'Mean_Train_Acc': mean_metrics['Train Acc (%)'].values,
        'Std_Train_Acc': std_metrics['Train Acc (%)'].values,
        'Mean_Test_Acc': mean_metrics['Test Acc (%)'].values,
        'Std_Test_Acc': std_metrics['Test Acc (%)'].values,
        'Mean_Precision': mean_metrics['Precision'].values,
        'Std_Precision': std_metrics['Precision'].values,
        'Mean_F1': mean_metrics['F1'].values,
        'Std_F1': std_metrics['F1'].values,
        'Mean_AUC': mean_metrics['AUC'].values,
        'Std_AUC': std_metrics['AUC'].values,
        'Mean_Total_Time': mean_metrics['Total Time (s)'].values,
        'Std_Total_Time': std_metrics['Total Time (s)'].values,
        'Mean_Peak_Memory_Usage': mean_metrics['Avg Memory Usage (MB)'].values,
        'Std_Peak_Memory_Usage': std_metrics['Avg Memory Usage (MB)'].values,
    })

    # Save regular results
    mean_results.to_csv(output_csv, index=False)

    # Build formatted version
    formatted_summary = pd.DataFrame()
    for idx, row in mean_results.iterrows():
        formatted_row = {}
        for key in mean_results.columns:
            if key.startswith("Mean_"):
                metric_name = key.replace("Mean_", "")
                std_key = "Std_" + metric_name
                mean_val = row[key]
                std_val = row[std_key]
                formatted_row[metric_name] = f"{mean_val:.3f} ± {std_val:.3f}"
        formatted_summary = pd.concat([formatted_summary, pd.DataFrame([formatted_row])], ignore_index=True)

    # Save the formatted version
    formatted_output_csv = os.path.splitext(output_csv)[0] + "_formatted.csv"
    formatted_summary.to_csv(formatted_output_csv, index=False)
    print(f"Cross-validation results".center(60,"-"))
    print(mean_results)
    print(f"Mean+std:")
    print(formatted_summary)
    return mean_results