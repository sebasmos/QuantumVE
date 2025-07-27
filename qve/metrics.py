import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import joblib
from tabulate import tabulate

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    f1_score,
    roc_auc_score
)
def get_metrics_multiclass_case_test(trained_svc, kernel_valid, Y_val_sub, peak_memory_usage, n_dim, data_train, data_test, total_time, output_dir):
    acc_test = trained_svc.score(kernel_valid, Y_val_sub)
    y_pred = trained_svc.predict(kernel_valid)
    y_probs = trained_svc.predict_proba(kernel_valid)

    precision = precision_score(Y_val_sub, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(Y_val_sub, y_pred, average="weighted", zero_division=1)
    print(f"Y_val_sub: ", set(Y_val_sub))
    print(f"y_pred: ", set(y_pred))
    print("classes_ learned by the classifier:", trained_svc.classes_)
    full_labels = list(range(10))
    cm = confusion_matrix(Y_val_sub, y_pred, labels=full_labels)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(l) for l in full_labels])
    disp.plot(cmap=plt.cm.Blues)
    # plt.title("MNIST Test Confusion Matrix")

    # More informative PNG filename
    cm_filename = os.path.join(
        output_dir,
        f"conf_matrix_test_qubits{n_dim}_train{len(data_train)}_test{len(data_test)}_f1{f1:.3f}.png"
    )
    plt.savefig(cm_filename)
    plt.show(block=True)
    print(f"✅ Confusion matrix saved to: {cm_filename}")
    
    auc = float(roc_auc_score(Y_val_sub, y_probs, multi_class='ovr', average='weighted', labels=full_labels))
    
    results = {
        "n_dim": n_dim,
        "Test Size": len(data_test),
        "Test Acc (%)": round(acc_test, 3),
        "Precision": round(precision, 3),
        "F1": round(f1, 3),
        "AUC": round(auc, 3),
        "Total Time (s)": round(total_time, 3),
        "Peak Memory Usage (MB)": float(round(peak_memory_usage, 3)),
    }

    # More informative model filename
    model_filename = os.path.join(
        output_dir,
        f"quantum_svm_test_model_qubits{n_dim}_train{len(data_train)}_test{len(data_test)}_f1{f1:.3f}.pkl"
    )
    joblib.dump(trained_svc, model_filename)
    print(f"✅ Model saved to: {model_filename}")
    print(results)

    return results
    

def get_metrics_multiclass_case(trained_svc, kernel_train, Y_train_sub, kernel_valid, Y_val_sub,
                                training_time, peak_memory_usage, n_dim, data_train, data_test, exp_t,
                                oper_t, path_t, tnsm_kernel_t, partition_ratio, train_amp_count,
                                valid_amp_count, output_dir):
    acc_train = trained_svc.score(kernel_train, Y_train_sub)
    acc_test = trained_svc.score(kernel_valid, Y_val_sub)
    y_pred = trained_svc.predict(kernel_valid)
    y_probs = trained_svc.predict_proba(kernel_valid)

    precision = precision_score(Y_val_sub, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(Y_val_sub, y_pred, average="weighted", zero_division=1)

    full_labels = list(np.unique(Y_train_sub))
    cm = confusion_matrix(Y_val_sub, y_pred, labels=full_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(l) for l in full_labels])
    disp.plot(cmap=plt.cm.Blues)
    # plt.title("MNIST CV Fold Confusion Matrix")

    cm_filename = os.path.join(
        output_dir,
        f"conf_matrix_cv_qubits{n_dim}_train{len(data_train)}_test{len(data_test)}_f1{f1:.3f}.png"
    )
    plt.savefig(cm_filename)
    plt.show(block=True)

    auc = roc_auc_score(Y_val_sub, y_probs, multi_class='ovr', average='weighted', labels=full_labels)

    results = {
        "n_dim": n_dim,
        "Train Size": len(data_train),
        "Test Size": len(data_test),
        "Train Acc (%)": round(acc_train, 3),
        "Test Acc (%)": round(acc_test, 3),
        "Precision": round(precision, 3),
        "F1": round(f1, 3),
        "AUC": round(auc, 3),
        "Total Time (s)": round(training_time, 3),
        "Peak Memory Usage (MB)": round(peak_memory_usage, 3),
    }

    model_filename = os.path.join(
        output_dir,
        f"quantum_svm_cv_model_qubits{n_dim}_train{len(data_train)}_test{len(data_test)}_f1{f1:.3f}.pkl"
    )
    joblib.dump(trained_svc, model_filename)

    return results
def get_metrics_multiclass_case_cv(trained_svc, kernel_train, Y_train_sub, kernel_valid, Y_val_sub,
                                   peak_memory_usage, n_dim, data_train, total_time, fold=None, config=None):
    acc_train = trained_svc.score(kernel_train, Y_train_sub)
    acc_test = trained_svc.score(kernel_valid, Y_val_sub)

    y_pred = trained_svc.predict(kernel_valid)
    y_probs = trained_svc.predict_proba(kernel_valid)

    precision = precision_score(Y_val_sub, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(Y_val_sub, y_pred, average="weighted", zero_division=1)

    full_labels = list(np.unique(Y_train_sub))
    cm = confusion_matrix(Y_val_sub, y_pred, labels=full_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(l) for l in full_labels])
    disp.plot(cmap=plt.cm.Blues)
    # plt.title(f"MNIST Confusion Matrix - Fold {fold}")

    cm_filename = os.path.join(
        config["results"],
        f"conf_matrix_cv_fold{fold}_qubits{n_dim}_train{len(data_train)}_f1{f1:.3f}.png"
    )
    plt.savefig(cm_filename)
    plt.show(block=True)

    auc = roc_auc_score(Y_val_sub, y_probs, multi_class='ovr', average='weighted', labels=full_labels)

    results = {
        "Fold": fold,
        "n_dim": n_dim,
        "Train Size": len(data_train),
        "Val Size": len(Y_val_sub),
        "Train Acc (%)": round(acc_train, 3),
        "Test Acc (%)": round(acc_test, 3),
        "Precision": round(precision, 3),
        "F1": round(f1, 3),
        "AUC": round(auc, 3),
        "Total Time (s)": round(total_time, 3),
        "Avg Memory Usage (MB)": round(peak_memory_usage, 3)
    }

    return results