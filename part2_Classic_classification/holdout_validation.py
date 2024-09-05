import os
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import argparse
import sys
import random
import pickle
import os
# export PYTHONPATH="../quantumVM:$PYTHONPATH"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from qnet import *


def main(args):
    for seed in range(args.total_num_seed):
        print(f'SEED {seed}')
        print('--------------------------------')
        device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

        train_csv = os.path.join(args.data_path, 'train_embeddings.csv')
        val_csv = os.path.join(args.data_path, 'val_embeddings.csv')

        experiment_folder = f"{args.model}_{args.data_path}"
        
        if args.pca:
            experiment_folder += f"_PCA_{args.variance_threshold}"
            
        seed_dir = os.path.join(f"./{args.output_dir}", experiment_folder, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        train_dataset = EmbeddingDataset(train_csv, shuffle=True)
        val_dataset = EmbeddingDataset(val_csv, shuffle=False)
        
        X_train, y_train = train_dataset.features, train_dataset.labels
        X_val, y_val = val_dataset.features, val_dataset.labels
        
        if args.pca:
            X_train, pca_model = apply_pca(X_train, variance_threshold=args.variance_threshold)
            X_val = pca_model.transform(X_val)
            
            print("PCA shapes: ", X_train.shape, X_val.shape)

            # Concatenate the labels as the last column
            X_train_with_labels = np.hstack([X_train, y_train.reshape(-1, 1)])
            X_val_with_labels = np.hstack([X_val, y_val.reshape(-1, 1)])


            np.savetxt(os.path.join(seed_dir, f'train_reducedPCA_{args.variance_threshold}.csv'), X_train_with_labels, delimiter=',')
            np.savetxt(os.path.join(seed_dir, f'val_reducedPCA_{args.variance_threshold}.csv'), X_val_with_labels, delimiter=',')

            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        
        class_counts = np.bincount(np.asarray(y_train, int))
        class_weights = 1.0 / class_counts
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        patience = 50  
        best_f1 = 0.0
        patience_counter = 0
        
        if args.model == 'svc':

            # best_params = {'C': 3, 'gamma': 0.01, 'kernel': 'rbf'}
            best_params = {'C': 10, 'gamma': "scale", 'kernel': 'rbf'}
            model = SVC(**best_params, probability=True, random_state=seed)
            
            start_train_time = time.time()
            model.fit(X_train, y_train)
            end_train_time = time.time()

            model_memory = get_memory_usage()

            start_inference_time = time.time()

            y_proba = model.predict_proba(X_val)

            y_pred = np.argmax(y_proba, axis=1)

            end_inference_time = time.time()

            training_time = end_train_time - start_train_time
            inference_time = end_inference_time - start_inference_time
            all_labels, all_preds = y_val, y_pred
            
        elif args.model == 'mlp':
            torch.manual_seed(seed)
            np.random.seed(seed)
            class_weights = class_weights.to(device)
            num_columns = X_train.shape[1]#number of columns
            # model = Classifier(input_dim=num_columns, hidden_units=256, num_classes=args.num_classes, dropout_rate=0.5).to(device)
            # model = SClassifier(num_columns, args.num_classes, hidden_sizes=[128, 64]).to(device)
            model = SClassifier(num_columns, args.num_classes, hidden_sizes=[256, 128, 64]).to(device)
            model.apply(reset_weights)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            
            scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.8)

            train_losses = []
            val_losses = []
            epoch_train_loss,epoch_val_loss = 0,0

            start_train_time = time.time()
            
            for epoch in range(args.num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device) 
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item() * inputs.size(0)
                epoch_train_loss /= len(train_loader.dataset)
                train_losses.append(epoch_train_loss)
                model.eval()
                all_labels, all_preds = [], []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        
                        inputs, labels = inputs.to(device), labels.to(device) 
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        epoch_val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        all_labels.extend(labels.cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())
                        

                epoch_val_loss /= len(val_loader.dataset)
                val_losses.append(epoch_val_loss)

                f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                print(f'Epoch {epoch+1}/{args.num_epochs}, F1 Score: {f1}')
                scheduler.step(f1)
                if f1 > best_f1:
                    best_f1 = f1
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(seed_dir, 'best_model.pth'))
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

            end_train_time = time.time()
            model_memory = get_memory_usage()
            start_inference_time = time.time()
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device) 
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
            end_inference_time = time.time()

            training_time = end_train_time - start_train_time
            inference_time = end_inference_time - start_inference_time
            plot_losses(train_losses, val_losses, seed_dir)                   

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_075 = fbeta_score(all_labels, all_preds, beta=0.75, average='weighted', zero_division=0)
        
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
        save_predictions(args, experiment_folder, seed, all_labels, all_preds, model, metrics)
        plot_multiclass_roc_curve(all_labels, all_preds, seed_dir)
        
        save_confusion_matrix(all_labels, all_preds, np.unique(y_train),seed_dir, "val")
    consolidate_and_average_metrics(args, experiment_folder, os.path.join(f"./{args.output_dir}", experiment_folder))    
    print("Evaluation across different seeds completed. Results saved to the output directory.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a simple MLP or SVC with K-Fold Cross Validation")
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset directory containing CSV files')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output metrics and models')
    parser.add_argument('--total_num_seed', type=int, default=20, help='seed')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs (only applicable for MLP)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (only applicable for MLP)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer (only applicable for MLP)')
    parser.add_argument('--model', type=str, choices=['mlp', 'svc'], default='mlp', help='Model to use: "mlp" or "svc"')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cpu", "cuda")')
    parser.add_argument('--pca', action='store_true', help='Perform evaluation only')
    parser.add_argument('--variance_threshold', type=float, default=0.8, help='Batch size for training (only applicable for MLP)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
