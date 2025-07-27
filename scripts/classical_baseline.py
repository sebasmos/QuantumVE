import os
import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
import joblib

def main():
    parser = argparse.ArgumentParser(description="Run SVC with sklearn's CV and test evaluation")
    parser.add_argument("--data_path", type=str, default="../data/fashionmnist_embeddings/vit-l_14_at_336px_768")
    parser.add_argument("--output_dir", type=str, default="../EXPERIMENTS/outputs")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--total_train_samples", type=int, default=10000)
    parser.add_argument("--total_test_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and shuffle
    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))

    train_df = shuffle(train_df, random_state=args.seed)
    test_df = shuffle(test_df, random_state=args.seed)

    X = train_df.iloc[:, :-1].values[:args.total_train_samples]
    Y = train_df.iloc[:, -1].values[:args.total_train_samples]

    X_test = test_df.iloc[:, :-1].values[-args.total_test_samples:]
    Y_test = test_df.iloc[:, -1].values[-args.total_test_samples:]

    print(f"Training on {X.shape[0]} samples with {args.cv}-fold cross-validation...")
    print(f"Testing on {X_test.shape[0]} samples...")

    model = SVC(probability=True)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    scores = cross_validate(
        model, X, Y, cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "roc_auc_ovr": "roc_auc_ovr"
        },
        return_train_score=True
    )

    df = pd.DataFrame(scores)
    df.to_csv(os.path.join(args.output_dir, "svc_sklearn_cv_results.csv"), index=False)
    print("\nCross-validation results:")
    print(df.mean().round(4))

    # Train final model and evaluate on test set
    best_model = SVC(kernel="rbf", probability=True)
    best_model.fit(X, Y)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    test_metrics = {
        "test_accuracy": accuracy_score(Y_test, y_pred),
        "test_f1": f1_score(Y_test, y_pred, average="weighted"),
        "test_precision": precision_score(Y_test, y_pred, average="weighted"),
        "test_roc_auc_ovr": roc_auc_score(Y_test, y_proba, multi_class="ovr")
    }

    print("\nHeld-out test results:")
    print({k: round(v, 4) for k, v in test_metrics.items()})

    pd.DataFrame([test_metrics]).to_csv(os.path.join(args.output_dir, "svc_test_results.csv"), index=False)
    joblib.dump(best_model, os.path.join(args.output_dir, "svc_final_model.pkl"))

if __name__ == "__main__":
    main()
