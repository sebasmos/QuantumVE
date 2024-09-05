from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
def compute_f0_75_score_mean(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    beta = 0.75
    f0_75_scores = []
    for p, r in zip(precision, recall):
        if p + r == 0:
            f0_75 = 0.0
        else:
            f0_75 = (1 + beta**2) * (p * r) / (beta**2 * p + r)
        f0_75_scores.append(f0_75)
    
    return np.mean(f0_75_scores)