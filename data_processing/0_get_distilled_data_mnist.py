import os
os.environ["CUQUANTUM_LOG_LEVEL"] = "OFF"
import argparse
import logging
import os
import time
from itertools import combinations, chain, product
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, f1_score, roc_auc_score, accuracy_score
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from mpi4py import MPI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tabulate import tabulate

from cuquantum import *
import cupy as cp
from cupy.cuda.runtime import getDeviceCount
from sklearn.datasets import load_digits, fetch_openml
from functools import cache

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image

# ---------------------- Function Definitions ----------------------

def balanced_kmeans_distillation(X, Y, k_per_class):
        distilled_X, distilled_Y, distilled_indices = [], [], []
        for label in np.unique(Y):
            X_class = X[Y == label]
            Y_class = Y[Y == label]
            indices_class = np.where(Y == label)[0]
            kmeans = KMeans(n_clusters=k_per_class, random_state=42, n_init="auto")
            kmeans.fit(X_class)
            for center in kmeans.cluster_centers_:
                idx = np.argmin(np.linalg.norm(X_class - center, axis=1))
                distilled_X.append(X_class[idx])
                distilled_Y.append(Y_class[idx])
                distilled_indices.append(indices_class[idx])
        return np.array(distilled_X), np.array(distilled_Y), np.array(distilled_indices)

def main():
    parser = argparse.ArgumentParser(description="Quantum SVM Training with TN-SM and Distillation")

    parser.add_argument("--output_dir", type=str, default="../data/mnist_embeddings/",
                        help="Directory to save experiments and results")
    parser.add_argument("--max", type=int, default=70000,
                        help="Maximum number of distilled samples to consider")
    parser.add_argument("--k_per_class", type=int, default=200,
                        help="Number of distilled samples per class for KMeans distillation")
    parser.add_argument("--qubits", type=int, default=2,
                        help="Number of qubits / PCA components")
    parser.add_argument("--train_samples", type=int, default=1600,
                        help="Exact number of training samples to use")
    parser.add_argument("--test_samples", type=int, default=400,
                        help="Exact number of testing samples to use")
    parser.add_argument("--seed", type=int, default=255,
                        help="Random seed for shuffling distilled data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_baseline = os.path.join(args.output_dir, "baseline")
    os.makedirs(save_baseline, exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(root=f"{args.output_dir}/raw", train=True, download=True, transform=transform)
    images_tensor = dataset.data.float() / 255.0
    images_np = images_tensor.numpy()
    labels = dataset.targets.numpy().astype(int)
    flattened_images = images_np.reshape(len(images_np), -1)

    assert len(flattened_images) == 60000

    from torchvision.datasets import MNIST as MNIST_TEST
    test_dataset = MNIST_TEST(root=f"{args.output_dir}/raw", train=False, download=True, transform=transform)
    test_images_tensor = test_dataset.data.float() / 255.0
    test_images_np = test_images_tensor.numpy()
    test_labels = test_dataset.targets.numpy().astype(int)
    test_flattened_images = test_images_np.reshape(len(test_images_np), -1)

    full_images_np = np.concatenate([images_np, test_images_np], axis=0)
    full_flattened = np.concatenate([flattened_images, test_flattened_images], axis=0)
    full_labels = np.concatenate([labels, test_labels], axis=0)

    assert full_flattened.shape[0] == 70000

    X_distilled, Y_distilled, indices_distilled = balanced_kmeans_distillation(
        full_flattened, full_labels, args.k_per_class
    )

    np.random.seed(args.seed)
    perm = np.random.permutation(len(X_distilled))
    X_shuffled = X_distilled[perm]
    Y_shuffled = Y_distilled[perm]
    indices_shuffled = indices_distilled[perm]

    total_required = args.train_samples + args.test_samples
    data_train = X_shuffled[:args.train_samples]
    Y_train = Y_shuffled[:args.train_samples]
    train_indices = indices_shuffled[:args.train_samples]

    data_test = X_shuffled[args.train_samples:total_required]
    Y_test = Y_shuffled[args.train_samples:total_required]
    test_indices = indices_shuffled[args.train_samples:total_required]

    # --- Save CSV files with proper format using pandas ---

    
    # Train
    df_train = pd.DataFrame(data_train)
    df_train["label"] = Y_train
    df_train.to_csv(os.path.join(save_baseline, f"train.csv"), index=False)

    # Test
    df_test = pd.DataFrame(data_test)
    df_test["label"] = Y_test
    df_test.to_csv(os.path.join(save_baseline, f"test.csv"), index=False)

    # Indices
    pd.DataFrame(train_indices, columns=["index"]).to_csv(
        os.path.join(save_baseline, f"train_indices_{df_train.shape[1]-1}.csv"), index=False
    )
    pd.DataFrame(test_indices, columns=["index"]).to_csv(
        os.path.join(save_baseline, f"test_indices_{df_train.shape[1]-1}.csv"), index=False
    )

    # Full data
    # pd.DataFrame(full_flattened).to_csv(
    #     os.path.join(args.output_dir, f"full_flattened_{df_train.shape[1]-1}.csv"), index=False
    # )
    # pd.DataFrame(full_labels, columns=["label"]).to_csv(
    #     os.path.join(args.output_dir, f"full_labels_{df_train.shape[1]-1}.csv"), index=False
    # )

    # Print summary
    print("Training data shape:", data_train.shape)
    print("Unique values in training labels:", np.unique(Y_train))

    # --- Save distilled images based on original indexes ---
    train_img_dir = os.path.join(args.output_dir, "distilled_train_images")
    test_img_dir = os.path.join(args.output_dir, "distilled_test_images")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    for label, idx in zip(Y_train, train_indices):
        img = Image.fromarray((full_images_np[idx] * 255).astype(np.uint8))
        img.save(os.path.join(train_img_dir, f"{label}_{idx}.png"))

    for label, idx in zip(Y_test, test_indices):
        img = Image.fromarray((full_images_np[idx] * 255).astype(np.uint8))
        img.save(os.path.join(test_img_dir, f"{label}_{idx}.png"))

    print(f"Saved distilled training images to: {train_img_dir}")
    print(f"Saved distilled testing images to: {test_img_dir}")

if __name__ == "__main__":
    main()
