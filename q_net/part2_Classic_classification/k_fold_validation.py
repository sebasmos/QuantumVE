import os
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import psutil
import argparse
import sys
# export PYTHONPATH="../quantumVM:$PYTHONPATH"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from qnet.graphics import *
from qnet.metrics import *
from qnet.memory import *
from qnet.embeddings import *
from qnet.model import *
from qnet.utils import *

"""
python k_fold_validation.py --data_path efficientnet_b3_1536_bs64  --num_epochs 50 --model svc
"""

def reset_weights(m):
    '''
    Try resetting model weights to avoid weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def save_metrics(output_dir, fold, metrics):
    df = pd.DataFrame(metrics, index=[0])
    df.to_csv(os.path.join(output_dir, f'metrics_fold_{fold}.csv'), index=False)

def save_split_data(output_dir, fold, train_idx, test_idx, dataset):
    split_dir = os.path.join(output_dir, f'split_k_{fold}')
    os.makedirs(split_dir, exist_ok=True)
    train_data = dataset.data.iloc[train_idx]
    train_data.to_csv(os.path.join(split_dir, 'train_split.csv'), index=False)
    test_data = dataset.data.iloc[test_idx]
    test_data.to_csv(os.path.join(split_dir, 'test_split.csv'), index=False)

def calculate_avg_metrics(output_dir, k_folds):
    metrics_list = []
    for fold in range(k_folds):
        split_dir = os.path.join(output_dir, f'split_k_{fold}')
        metrics_file = os.path.join(split_dir, f'metrics_fold_{fold}.csv')
        
        # Check if the metrics file exists before attempting to read it
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            metrics_list.append(df)
        else:
            print(f"Metrics file for fold {fold} not found at {metrics_file}. Skipping this fold.")
    if not metrics_list:
        print("No metrics files found. Cannot calculate average metrics.")
        return

    avg_metrics = pd.concat(metrics_list).mean().to_frame().T
    avg_metrics.to_csv(os.path.join(output_dir, 'average_metrics.csv'), index=False)

    print("Average metrics calculated and saved to 'average_metrics.csv'.")


def main(args):
    torch.manual_seed(42)
    
    train_csv = os.path.join(args.data_path, 'train_embeddings.csv')
    val_csv = os.path.join(args.data_path, 'val_embeddings.csv')
    
    train_dataset = EmbeddingDataset(train_csv)
    val_dataset = EmbeddingDataset(val_csv)
    
    kfold = KFold(n_splits=args.k_folds, shuffle=True)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(train_dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        split_dir = os.path.join(args.output_dir, f'split_k_{fold}')
        os.makedirs(split_dir, exist_ok=True)
        
        save_split_data(args.output_dir, fold, train_idx, test_idx, train_dataset)

        if args.model == 'mlp':
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
            test_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx))
            
            num_columns = train_dataset.data.shape[1] - 1
            model = SimpleMLP(input_dim=num_columns)
            model.apply(reset_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = nn.CrossEntropyLoss()

            start_train_time = time.time()
            for epoch in range(args.num_epochs):
                print(f"EPOCH {epoch}")
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            end_train_time = time.time()

            metrics = {
                'Training time': end_train_time - start_train_time,
                'Memory RAM': get_memory_usage()
            }

            model.eval()
            correct, total = 0, 0
            all_labels, all_preds = [], []
            
            start_inference_time = time.time()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
            end_inference_time = time.time()

            metrics['Inference time'] = end_inference_time - start_inference_time

        elif args.model == 'svc':
            X_train = train_dataset.features[train_idx]
            y_train = train_dataset.labels[train_idx]
            X_test = train_dataset.features[test_idx]
            y_test = train_dataset.labels[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            best_params = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}#{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
            model = SVC(**best_params, random_state=42,probability=True)#probability=True

            start_train_time = time.time()
            model.fit(X_train, y_train)
            end_train_time = time.time()

            metrics = {
                'Training time': end_train_time - start_train_time,
                'Memory RAM': get_memory_usage()
            }

            start_inference_time = time.time()
            y_pred = model.predict(X_test)
            end_inference_time = time.time()

            metrics['Inference time'] = end_inference_time - start_inference_time
            all_labels, all_preds = y_test, y_pred
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        f0_75 = compute_f0_75_score_mean(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        plot_multiclass_roc_curve(all_labels, all_preds, split_dir) 
        metrics.update({'Accuracy': accuracy, 'F1 Score': f1, 'F0.75 Score': f0_75, 'Precision': precision,
                        'Recall': recall, 'Training time': metrics['Training time'], 'Inference time': metrics['Inference time'], 
                        'Memory RAM': metrics['Memory RAM']})

        save_metrics(split_dir, fold, metrics)

        if args.model == 'mlp':
            torch.save(model.state_dict(), os.path.join(split_dir, f'model_fold_{fold}.pth'))

    calculate_avg_metrics(args.output_dir, args.k_folds)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    all_labels, all_preds = [], []

    if args.model == 'mlp':
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

    elif args.model == 'svc':
        all_inputs, all_labels = [], []
        for inputs, labels in val_loader:
            all_inputs.append(inputs.numpy())
            all_labels.extend(labels.cpu().numpy())
        
        all_inputs = np.vstack(all_inputs)  # Stack them along the first axis to get a single array
        all_preds = model.predict(all_inputs)
   
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds,  average='weighted')
    f0_75 = compute_f0_75_score_mean(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds,  average='weighted')
    recall = recall_score(all_labels, all_preds,  average='weighted')
    val_metrics = {'Accuracy': accuracy, 'F1 Score': f1, 'F0.75 Score': f0_75, 'Precision': precision,
                       'Recall': recall, 'Training time': metrics['Training time'], 'Inference time': metrics['Inference time'], 
                        'Memory RAM': metrics['Memory RAM']
                       }
    df = pd.DataFrame(val_metrics, index=[0])
    df.to_csv(os.path.join(args.output_dir, 'validation_metrics.csv'), index=False)

    print("K-Fold Cross Validation and testing completed. Results saved to the output directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a simple MLP or SVC with K-Fold Cross Validation")
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset directory containing CSV files')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output metrics and models')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of K-Folds for cross-validation')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs (only applicable for MLP)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (only applicable for MLP)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer (only applicable for MLP)')
    parser.add_argument('--model', type=str, choices=['mlp', 'svc'], default='mlp', help='Model to use: "mlp" or "svc"')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)