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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import argparse
import sys
import random
import pickle
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
import os
# export PYTHONPATH="../quantumVM:$PYTHONPATH"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from qnet.graphics import *
from qnet.metrics import *
from qnet.memory import *
from qnet.embeddings import *
from qnet.model import *
from qnet.utils import *




def main(args):
    for seed in range(args.total_num_seed):
        print(f'SEED {seed}')
        print('--------------------------------')

        train_csv = os.path.join(args.data_path, 'train_embeddings.csv')
        val_csv = os.path.join(args.data_path, 'val_embeddings.csv')
        
        seed_dir = os.path.join(f"./{args.output_dir}", f"{args.model}_{args.data_path}", f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        train_dataset = EmbeddingDataset(train_csv, shuffle=True)  # Shuffle data
        val_dataset = EmbeddingDataset(val_csv, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        if args.model == 'mlp':
            torch.manual_seed(seed)
            np.random.seed(seed)
            num_columns = train_dataset.data.shape[1] - 1
            model = Classifier(input_dim=num_columns, hidden_units=256, num_classes=6, dropout_rate=0.5)
            model.apply(reset_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = nn.CrossEntropyLoss()

            start_train_time = time.time()
            for epoch in range(args.num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            end_train_time = time.time()

            model_memory = get_memory_usage()

            model.eval()
            all_labels, all_preds = [], []
            
            start_inference_time = time.time()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
            end_inference_time = time.time()

            training_time = end_train_time - start_train_time
            inference_time = end_inference_time - start_inference_time

        elif args.model == 'svc':

            X_train, y_train = train_dataset.features, train_dataset.labels
            X_test, y_test = val_dataset.features, val_dataset.labels
            
            best_params = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
            model = SVC(**best_params, probability=True, random_state=seed)
            
            start_train_time = time.time()
            model.fit(X_train, y_train)
            end_train_time = time.time()

            model_memory = get_memory_usage()

            start_inference_time = time.time()

            y_proba = model.predict_proba(X_test)

            y_pred = np.argmax(y_proba, axis=1)

            end_inference_time = time.time()

            training_time = end_train_time - start_train_time
            inference_time = end_inference_time - start_inference_time
            all_labels, all_preds = y_test, y_pred

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1_075 = compute_f0_75_score_mean(all_labels, all_preds)

        metrics = {
            'Accuracy': accuracy,
            'F1 Score': f1,
            'F0.75 Score': f1_075,
            'Precision': precision,
            'Recall': recall,
            'Training time': training_time,
            'Inference time': inference_time,
            'Memory RAM': model_memory
        }
        print(metrics)
        save_predictions(args, f"{args.model}_{args.data_path}", seed, all_labels, all_preds, model, metrics)

    consolidate_and_average_metrics(args)
    print("Evaluation across different seeds completed. Results saved to the output directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a simple MLP or SVC with K-Fold Cross Validation")
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset directory containing CSV files')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output metrics and models')
    parser.add_argument('--total_num_seed', type=int, default=3, help='seed')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs (only applicable for MLP)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (only applicable for MLP)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer (only applicable for MLP)')
    parser.add_argument('--model', type=str, choices=['mlp', 'svc'], default='mlp', help='Model to use: "mlp" or "svc"')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
