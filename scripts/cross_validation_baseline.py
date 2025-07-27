# Section 3: Cross-Validation

# Built-in and system
import os
import sys
import time
import argparse
import collections
import importlib
import pickle

# Third-party libraries
import numpy as np
import pandas as pd
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import sympy
import torch
import joblib
from tabulate import tabulate
from memory_profiler import memory_usage
from mpi4py import MPI

# Scikit-learn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_digits, fetch_openml

# Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap

# cuQuantum and CUDA
from cuquantum import *
from cupy.cuda import nccl
from cupy.cuda.runtime import getDeviceCount

from qve import set_seed, process_folds, get_metrics_multiclass_case_cv, get_metrics_multiclass_case_test, data_prepare_cv

# Utilities
from itertools import combinations, chain, product

# Reload pkg_resources if needed
import pkg_resources
importlib.reload(pkg_resources)

gpu_count = torch.cuda.device_count()
print(f"Number of GPUs detected: {gpu_count}")

if gpu_count > 0:
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU detected.")

# ---------------------- Function Definitions ----------------------

def make_bsp(n_dim):
    param = ParameterVector("p",n_dim)
    bsp_qc = QuantumCircuit(n_dim)
    bsp_qc.h(list(range(n_dim)))
    i = 0
    for q in range(n_dim):
        bsp_qc.rz(param.params[q],[q])
        bsp_qc.ry(param.params[q],[q])
    for q in range(n_dim-1):
        bsp_qc.cx(0+i, 1+i)
        i+=1
    for q in range(n_dim):
        bsp_qc.rz(param.params[q],[q])
    return bsp_qc

def build_qsvm_qc(bsp_qc, n_dim, y_t, x_t):
    """Builds a quantum circuit for the quantum SVM kernel."""
    qc_1 = bsp_qc.assign_parameters(y_t).to_gate()
    qc_2 = bsp_qc.assign_parameters(x_t).inverse().to_gate()
    kernel_qc = QuantumCircuit(n_dim)
    kernel_qc.append(qc_1, list(range(n_dim)))
    kernel_qc.append(qc_2, list(range(n_dim)))
    return kernel_qc

def renew_operand(n_dim,oper_tmp,y_t,x_t):
    oper = oper_tmp.copy()
    n_zg, n_zy_g = [], []
    for d1 in y_t:
        z_g  = np.array([[np.exp(-1j*0.5*d1),0],[0,np.exp(1j*0.5*d1)]])
        n_zg.append(z_g)
        y_g  = np.array([[np.cos(d1/2),-np.sin(d1/2)],[np.sin(d1/2),np.cos(d1/2)]])
        n_zy_g.append(z_g)
        n_zy_g.append(y_g)
    oper[n_dim*2:n_dim*4] = cp.array(n_zy_g)
    oper[n_dim*5-1:n_dim*6-1] = cp.array(n_zg)
    n_zgd, n_zy_gd = [], []
    for d2 in x_t[::-1]:       
        z_gd  = np.array([[np.exp(1j*0.5*d2),0],[0,np.exp(-1j*0.5*d2)]])
        n_zgd.append(z_gd)  
        y_gd  = np.array([[np.cos(d2/2),np.sin(d2/2)],[-np.sin(d2/2),np.cos(d2/2)]])
        n_zy_gd.append(y_gd)
        n_zy_gd.append(z_gd)
    oper[n_dim*6-1:n_dim*7-1] = cp.array(n_zgd)
    oper[n_dim*8-2:n_dim*10-2] = cp.array(n_zy_gd)
    return oper


def data_partition(indices_list,size,rank):
    num_data = len(indices_list)
    chunk, extra = num_data // size, num_data % size
    data_begin = rank * chunk + min(rank, extra)
    data_end = num_data if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    data_index = range(data_begin,data_end)
    indices_list_rank = indices_list[data_begin:data_end]
    return indices_list_rank

def data_to_operand(n_dim,operand_tmp,data1,data2,indices_list):
    operand_list = []
    for i1, i2 in indices_list:
        n_op = renew_operand(n_dim,operand_tmp,data1[i1-1],data2[i2-1])
        operand_list.append(n_op) 
    return operand_list


def operand_to_amp(opers, network):
    amp_tmp = []
    with network as tn:
        for i in range(len(opers)):
            tn.reset_operands(*opers[i])     
            amp_tn = abs(tn.contract()) ** 2
            amp_tmp.append(amp_tn)
    return amp_tmp
    

def get_kernel_matrix(data1, data2, amp_data, indices_list, mode=None):
    amp_m = list(chain.from_iterable(amp_data))
    # print(len(amp),len(indices_list))
    kernel_matrix = np.zeros((len(data1),len(data2)))
    i = -1
    for i1, i2 in indices_list:
        i += 1
        kernel_matrix[i1-1][i2-1] = np.round(amp_m[i],8)
    if mode == 'train':
        kernel_matrix = kernel_matrix + kernel_matrix.T+np.diag(np.ones((len(data2))))
    return kernel_matrix

def run_tnsm_cv(data_train, data_val, Y_train_sub, Y_val_sub, n_dim,
                device_id, comm_mpi, rank, size, config, fold=None):
    start_time = time.time()
    list_train = list(combinations(range(1, len(data_train) + 1), 2))
    list_val = list(product(range(1, len(data_val) + 1), range(1, len(data_train) + 1)))
    list_train_partition = data_partition(list_train, size, rank)
    list_val_partition = data_partition(list_val, size, rank)

    t0 = time.time()
    bsp_qc = make_bsp(n_dim)
    circuit = build_qsvm_qc(bsp_qc, n_dim, data_train[0], data_train[0])
    converter = CircuitToEinsum(circuit, dtype='complex128', backend='cupy')
    a = str(0).zfill(n_dim)
    exp, oper = converter.amplitude(a)
    exp_t = round((time.time() - t0), 3)

    t0 = time.time()
    oper_train = data_to_operand(n_dim, oper, data_train, data_train, list_train_partition)
    oper_val = data_to_operand(n_dim, oper, data_val, data_train, list_val_partition)
    oper_t = round((time.time() - t0), 3)

    t0 = time.time()
    options = NetworkOptions(blocking="auto", device_id=device_id)
    network = Network(exp, *oper, options=options)
    path, info = network.contract_path()
    network.autotune(iterations=20)
    path_t = round((time.time() - t0), 3)

    t0 = time.time()
    oper_data = oper_train + oper_val
    amp_list = operand_to_amp(oper_data, network)
    amp_train = cp.array(amp_list[:len(oper_train)])
    amp_valid = cp.array(amp_list[len(oper_train):len(oper_train) + len(oper_val)])
    amp_data_train = comm_mpi.gather(amp_train, root=0)
    amp_data_valid = comm_mpi.gather(amp_valid, root=0)
    tnsm_kernel_t = round((time.time() - t0), 3)

    print("Train data shape", Y_train_sub.shape)
    print("Valid data shape", Y_val_sub.shape)

    if rank == 0:
        kernel_train = get_kernel_matrix(data_train, data_train, amp_data_train, list_train, mode='train')
        kernel_valid = get_kernel_matrix(data_val, data_train, amp_data_valid, list_val)

        def train_model():
            trained_svc = SVC(kernel="precomputed", probability=True)
            trained_svc.fit(kernel_train, Y_train_sub)
            return trained_svc

        mem_usage, trained_svc = memory_usage((train_model, ()), interval=0.1, retval=True)
        end_time = time.time()
        total_time = end_time - start_time
        peak_memory_usage = sum(mem_usage) / len(mem_usage)

        partition_ratio = round(len(list_train_partition) / len(list_train), 3)
        train_amp_count = len(amp_data_train)
        valid_amp_count = len(amp_data_valid)

        model_filename = os.path.join(config["results"], f"fold_{fold}_qubit_{n_dim}_Sample_{len(data_train)}_model.pkl")
        with open(model_filename, 'wb') as f:
            pickle.dump(trained_svc, f)

        print(f"✅ Model for Fold {fold} saved to {model_filename}")
        
        results = get_metrics_multiclass_case_cv(
            trained_svc, kernel_train, Y_train_sub, kernel_valid, Y_val_sub,
            peak_memory_usage, n_dim,  data_train, total_time, fold, config=config)

        results_df = pd.DataFrame([results])
        results_path = os.path.join(config["results"], f"{n_dim}_qubits_quantum_eval_metrics.csv")
        results_df.to_csv(results_path, mode='a', header=not os.path.exists(results_path), index=False)

        print(f"\n📊 Quantum SVM Results for Fold {fold}:\n")
        print(tabulate(results_df, headers="keys", tablefmt="grid"))
        
def compute_kernel_block(
    dtest,
    _dtrain,
    n_dim,
    device_id,
    make_bsp_fn,
    build_qsvm_qc_fn,
    CircuitToEinsum_cls,
    data_to_operand_fn,
    Network_cls,
    NetworkOptions_cls,
    operand_to_amp_fn,
    get_kernel_matrix_fn
):
    """Compute the kernel matrix for a block of data."""
    kernel_indices = list(product(range(1, len(dtest)+1), range(1, len(_dtrain)+1)))
    bsp_qc = make_bsp_fn(n_dim)
    circuit = build_qsvm_qc_fn(bsp_qc, n_dim, _dtrain[0], _dtrain[0])
    converter = CircuitToEinsum_cls(circuit, dtype='complex128', backend='cupy')
    a = str(0).zfill(n_dim)
    exp, oper = converter.amplitude(a)
    oper_test = data_to_operand_fn(n_dim, oper, dtest, _dtrain, kernel_indices)
    network = Network_cls(exp, *oper, options=NetworkOptions_cls(blocking="auto", device_id=device_id))
    network.contract_path()
    amp_test = operand_to_amp_fn(oper_test, network)
    amp_test = cp.asnumpy(cp.array(amp_test))
    kernel = get_kernel_matrix_fn(dtest, _dtrain, [amp_test], kernel_indices)
    return kernel

def main():
    parser = argparse.ArgumentParser(description="Quantum SVM Cross-Validation Setup")
    parser.add_argument("--data_path", type=str, default="../data/mnist_embeddings/baseline")
    parser.add_argument("--output_dir", type=str, default="../EXPERIMENTS/delete")
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--total_train_samples", type=int, default=100)
    parser.add_argument("--total_test_samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = vars(args)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    results = os.path.join(args.output_dir, f"qubits_{args.qubits}_CV")
    os.makedirs(results, exist_ok=True)
    config["results"] = results

    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    size = comm_mpi.Get_size()
    device_id = rank % getDeviceCount()
    cp.cuda.Device(device_id).use()

    ## LOAD DATA
    distilled_imgs_cv = os.path.join(args.data_path, "train.csv")
    distilled_imgs_test = os.path.join(args.data_path, "test.csv")
    
    train_df = pd.read_csv(distilled_imgs_cv)
    test_df = pd.read_csv(distilled_imgs_test)
    
    train_df = shuffle(train_df, random_state=args.seed)
    test_df = shuffle(test_df, random_state=args.seed)
    
    X_train = train_df.iloc[:, :-1].values  # all columns except last
    Y_train = train_df.iloc[:, -1].values   # last column (label)
    
    X_test = test_df.iloc[:, :-1].values
    Y_test = test_df.iloc[:, -1].values

    print(f"{'Loading data BEFORE processing':-^60}")
    print(f"X_train: ", X_train.shape)
    print(f"Y_train: ", Y_train.shape)

    print(f"X_test: ", X_test.shape)
    print(f"Y_test: ", Y_test.shape)

    data_cv, labels_cv = X_train[:args.total_train_samples], Y_train[:args.total_train_samples]
    data_test, labels_test = X_test[-args.total_test_samples:], Y_test[-args.total_test_samples:]
    
    qubits = args.qubits
    dcross_val, dtest = data_prepare_cv(qubits, data_cv, data_test)
    
    print(f"{'Loading data AFTER processing':-^60}")
    print(f"dcross_val: ", dcross_val.shape)

    print(f"dtest: ", dtest.shape)

    ## CV

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(dcross_val, labels_cv)):
        X_fold_train = dcross_val[train_idx]
        Y_fold_train = labels_cv[train_idx]
        X_fold_val = dcross_val[val_idx]
        Y_fold_val = labels_cv[val_idx]
        train_path = os.path.join(config["results"], f"train_fold_{fold+1}_qubits_{qubits}.csv")
        val_path = os.path.join(config["results"], f"val_fold_{fold+1}_qubits_{qubits}.csv")
        pd.DataFrame(X_fold_train).assign(label=Y_fold_train).to_csv(train_path, index=False)
        pd.DataFrame(X_fold_val).assign(label=Y_fold_val).to_csv(val_path, index=False)
        run_tnsm_cv(X_fold_train, X_fold_val, Y_fold_train, Y_fold_val, qubits, device_id, comm_mpi, rank, size, config=config, fold=fold + 1)
        
    if rank == 0:
        print("✅ Cross-validation complete.")
        input_csv = os.path.join(config["results"], f"{qubits}_qubits_quantum_eval_metrics.csv")
        output_csv = os.path.join(config["results"], f"mean_folds_metrics_{qubits}_qubits.csv")
        process_folds(args.folds, input_csv, output_csv)
        metrics_path = os.path.join(config["results"], str(qubits)+"_qubits_quantum_eval_metrics.csv")
        metrics_df = pd.read_csv(metrics_path)
        best_row = metrics_df.loc[metrics_df["Test Acc (%)"].idxmax()]
        best_fold_number = int(best_row["Fold"])
        best_acc = float(best_row["Test Acc (%)"])
        Train_Size = int(best_row["Train Size"])
        n_dim = int(best_row["n_dim"])
    
        print(f"{'Test on best k-fold model':-^60}")
        print(f"Best fold number : {best_fold_number}")
        print(f"Train size       : {Train_Size}")
        print(f"n_dim            : {n_dim}")
        print(f"Best Accuracy    : {best_acc}")
        print(f"Best model       : fold_{best_fold_number}_qubit_{n_dim}_Sample_{Train_Size}_model.pkl")
        
        train_path = os.path.join(config["results"], f"train_fold_{best_fold_number}_qubits_{n_dim}.csv")
        _dtrain = pd.read_csv(train_path).iloc[:,:-1].values.astype(np.float32)
        best_model_path = f"fold_{best_fold_number}_qubit_{n_dim}_Sample_{_dtrain.shape[0]}_model.pkl"
        with open(os.path.join(config["results"],best_model_path), 'rb') as f:
            trained_svc = pickle.load(f)
    
        train_save_path = os.path.join(config["results"], f"saved_train_data_qubits_{n_dim}.npz")
        np.savez(train_save_path, data=_dtrain)
    
        test_save_path = os.path.join(config["results"], f"saved_test_data_qubits_{n_dim}.npz")
        np.savez(test_save_path, data=dtest, labels=labels_test)

        start_time = time.time()
        mem_usage, kernel_test = memory_usage(
            (compute_kernel_block, (
                dtest,
                _dtrain,
                n_dim,
                device_id,
                make_bsp,
                build_qsvm_qc,
                CircuitToEinsum,
                data_to_operand,
                Network,
                NetworkOptions,
                operand_to_amp,
                get_kernel_matrix
            )),
            retval=True,
            interval=0.1
        )
        end_time = time.time()
        
        mem_used = max(mem_usage) - min(mem_usage)  # M
        mem_used = np.mean(mem_usage)  # MB
        total_time = end_time - start_time
    
        results = get_metrics_multiclass_case_test(
            trained_svc=trained_svc,
            kernel_valid=kernel_test,
            Y_val_sub=labels_test,
            peak_memory_usage=mem_used,
            n_dim=n_dim,
            total_time=total_time,
            data_train=_dtrain,
            data_test=dtest,
            output_dir=config["results"]
        )
        results_df = pd.DataFrame([results])
        results_path = os.path.join(config["results"], "test_metrics_heldout.csv")
        results_df.to_csv(results_path, mode='a', header=not os.path.exists(results_path), index=False)

if __name__ == "__main__":
    main()
