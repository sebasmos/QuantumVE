import os
import sys
import random
import pickle
import numpy as np 
import torch
import pandas as pd


def save_predictions(args, model_name, seed, y_test, y_pred, model, metrics):
    """
    Save predicted and true values as CSV files in a specified folder.

    Args:
    - pred_folder (str): The folder path where the files will be saved.
    - model_name (str): The name of the model.
    - seed (int): The seed used for randomization.
    - y_test (numpy array): True values.
    - y_pred (numpy array): Predicted values.
    """
    pred_folder = os.path.join(f"{args.output_dir}", model_name, f"seed_{seed}")

    os.makedirs(pred_folder, exist_ok=True)
    np.savetxt(os.path.join(pred_folder, 'y_test.csv'), y_test, delimiter=',', header="y_test", fmt='%d', comments='')
    np.savetxt(os.path.join(pred_folder, 'y_pred.csv'), y_pred, delimiter=',', header="y_pred", fmt='%d', comments='')


    df = pd.DataFrame(metrics, index=[0])
    df.to_csv(os.path.join(pred_folder, f'metrics_seed_{seed}.csv'), index=False)
    store_pickle_model(model, model_name, seed, pred_folder)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def store_pickle_model(model, model_name, seed, model_path):
    """
    Store a trained model as a pickle file.

    Args:
    - model: The trained model object to be stored.
    - model_name (str): The name of the model.
    - window_size (int): The window size used in the model.
    - model_path (str): The directory path where the model will be stored.
    """
    os.makedirs(model_path, exist_ok=True)

    model_name = f"{model_name}_{seed}"

    model_filename = os.path.join(model_path, model_name + '.pkl')

    print("Model saved in:", model_filename)
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"Failed to pickle the model due to: {e}")
