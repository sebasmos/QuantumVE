# -*- coding: utf-8 -*-

from __future__ import print_function
import torchvision.models as models
import torch
from urllib.request import urlopen
from PIL import Image
import timm

"""# Utility functions"""

import sys

sys.path.insert(0,'../')
sys.path.insert(0,'../../')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
# from MAE code
from util.datasets import build_dataset
import argparse
import util.misc as misc
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# import models_vit
import sys
import os
import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import torch; print(f'numpy version: {np.__version__}\nCUDA version: {torch.version.cuda} - Torch versteion: {torch.__version__} - device count: {torch.cuda.device_count()}')

from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
import torch
import pandas as pd
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
import numpy as np


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# Parameters
model_name = "eva02_large_patch14_448_embeddings_imageNet"#"mobilenetv4_r448" # or any other model name
batch_sizes = [8, 16, 32, 64]
embedding_sizes = [64,128,256,1024]
data_path = '../data/ABGQI_mel_spectrograms'
device = 'cuda'


MODEL_CONSTRUCTORS = {
    'eva02_large_patch14_448_embeddings_imageNet':timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True, num_classes=0),
    'mobilenetv4_r448_trained': timm.create_model('mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k', pretrained=False, num_classes=0),
    'mobilenetv4_r448': timm.create_model('mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k', pretrained=True, num_classes=0),
    'alexnet': models.alexnet,
    'convnext_base': models.convnext_base,
    'convnext_large': models.convnext_large,
    'convnext_small': models.convnext_small,
    'convnext_tiny': models.convnext_tiny,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b2': models.efficientnet_b2,
    'efficientnet_b3': models.efficientnet_b3,
    'efficientnet_b4': models.efficientnet_b4,
    'efficientnet_b5': models.efficientnet_b5,
    'efficientnet_b6': models.efficientnet_b6,
    'efficientnet_b7': models.efficientnet_b7,
    'efficientnet_v2_l': models.efficientnet_v2_l,
    'efficientnet_v2_m': models.efficientnet_v2_m,
    'efficientnet_v2_s': models.efficientnet_v2_s,
    'googlenet': models.googlenet,
    'inception_v3': models.inception_v3,
    'maxvit_t': models.maxvit_t,
    'mnasnet0_5': models.mnasnet0_5,
    'mnasnet0_75': models.mnasnet0_75,
    'mnasnet1_0': models.mnasnet1_0,
    'mnasnet1_3': models.mnasnet1_3,
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    'regnet_x_16gf': models.regnet_x_16gf,
    'regnet_x_1_6gf': models.regnet_x_1_6gf,
    'regnet_x_32gf': models.regnet_x_32gf,
    'regnet_x_3_2gf': models.regnet_x_3_2gf,
    'regnet_x_400mf': models.regnet_x_400mf,
    'regnet_x_800mf': models.regnet_x_800mf,
    'regnet_x_8gf': models.regnet_x_8gf,
    'regnet_y_128gf': models.regnet_y_128gf,# check this regnet_y_128gf: no weigthts avaialble
    'regnet_y_16gf': models.regnet_y_16gf,
    'regnet_y_1_6gf': models.regnet_y_1_6gf,
    'regnet_y_32gf': models.regnet_y_32gf,
    'regnet_y_3_2gf': models.regnet_y_3_2gf,
    'regnet_y_400mf': models.regnet_y_400mf,
    'regnet_y_800mf': models.regnet_y_800mf,
    'regnet_y_8gf': models.regnet_y_8gf,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnext101_32x8d': models.resnext101_32x8d,
    'resnext101_64x4d': models.resnext101_64x4d,
    'resnext50_32x4d': models.resnext50_32x4d,
    'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': models.shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': models.shufflenet_v2_x2_0,
    'squeezenet1_0': models.squeezenet1_0,
    'squeezenet1_1': models.squeezenet1_1,
    'swin_b': models.swin_b,
    'swin_s': models.swin_s,
    'swin_t': models.swin_t,
    'swin_v2_b': models.swin_v2_b,
    'swin_v2_s': models.swin_v2_s,
    'swin_v2_t': models.swin_v2_t,
    'vgg11': models.vgg11,
    'vgg11_bn': models.vgg11_bn,
    'vgg13': models.vgg13,
    'vgg13_bn': models.vgg13_bn,
    'vgg16': models.vgg16,
    'vgg16_bn': models.vgg16_bn,
    'vgg19': models.vgg19,
    'vgg19_bn': models.vgg19_bn,
    'vit_b_16': models.vit_b_16,
    'vit_b_32': models.vit_b_32,
    'vit_h_14': models.vit_h_14,# and this..no weigthts avaialble
    'vit_l_16': models.vit_l_16,
    'vit_l_32': models.vit_l_32,
    'wide_resnet101_2': models.wide_resnet101_2,
    'wide_resnet50_2': models.wide_resnet50_2
}


def count_parameters(model, message=""):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{message} Trainable params: {trainable_params} of {total_params}")

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def plot_multiclass_roc_curve(all_labels, all_predictions, EXPERIMENT_NAME="."):
    # Step 1: Label Binarization
    label_binarizer = LabelBinarizer()
    y_onehot = label_binarizer.fit_transform(all_labels)
    all_predictions_hot = label_binarizer.transform(all_predictions)

    # Step 2: Calculate ROC curves
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    unique_classes = range(y_onehot.shape[1])
    for i in unique_classes:
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], all_predictions_hot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Step 3: Plot ROC curves
    fig, ax = plt.subplots(figsize=(8, 8))

    # Micro-average ROC curve
    fpr_micro, tpr_micro, _ = roc_curve(y_onehot.ravel(), all_predictions_hot.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(
        fpr_micro,
        tpr_micro,
        label=f"micro-average ROC curve (AUC = {roc_auc_micro:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    # Macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in unique_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in unique_classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(unique_classes)
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)
    plt.plot(
        fpr_macro,
        tpr_macro,
        label=f"macro-average ROC curve (AUC = {roc_auc_macro:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    # Individual class ROC curves with unique colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    for class_id, color in zip(unique_classes, colors):
        plt.plot(
            fpr[class_id],
            tpr[class_id],
            color=color,
            label=f"ROC curve for Class {class_id} (AUC = {roc_auc[class_id]:.2f})",
            linewidth=2,
        )

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)  # Add diagonal line for reference
    plt.axis("equal")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\n to One-vs-Rest multiclass")
    plt.legend()
    plt.savefig(f'{EXPERIMENT_NAME}/roc_curve.png')
    plt.show()

def create_args(batch_size, model_name, embedding_size, output_dir, data_path, device):
    parser = argparse.ArgumentParser('VE extraction', add_help=False)
    parser.add_argument('--batch_size', default=batch_size, help='Batch size per GPU')
    parser.add_argument('--embedding_size', default=embedding_size, help='embedding_size')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations')
    parser.add_argument('--model', default=model_name, type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate')
    parser.add_argument('--data_path', default=data_path, type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=5, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default=output_dir,
                        help='path where to save')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default=device,
                        help='device to use for training/testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
        # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                            help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                            help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                            help='Label smoothing (default: 0.1)')
        # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                            help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                            help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                            help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                            help='Do not random erase first (clean) augmentation split')
        # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                            help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                            help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                            help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                            help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                            help='Probability of switching to cutmix when both mixup and cutmix enabled')

    parser.add_argument('--mixup_mode', type=str, default='batch',
                            help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--resume', default=".",
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=True, action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    args, unknown = parser.parse_known_args()
    return args

"""# Image Vector Embeddings Extraction"""

# # Define extract_embed class
class classifier_embeddings(nn.Module):
    def __init__(self, base_model, feat_space, model_name):
        super(classifier_embeddings, self).__init__()
        self.base_model = base_model
        self.model_name = model_name
        self.feat_space = feat_space
        # Example: Adding a new classifier layer
        if model_name in ("mobilenetv4_r448", "mobilenetv4_r448_trained"):
            self.new_classifier = nn.Linear(self.base_model.conv_head.out_channels, out_features=self.feat_space)
        elif model_name in ("eva02_large_patch14_448_embeddings_imageNet"):
            self.new_classifier = nn.Linear(in_features=1024, out_features=feat_space)
    def forward(self, x):
        x = self.base_model(x)
        x = self.new_classifier(x)
        return x

def load_and_initialize_model(model_name, weights_path, feat_space):
    model = timm.create_model('mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k', pretrained=False, num_classes=0)

    # Count parameters before loading the checkpoint
    count_parameters(model, message="Before loading checkpoint")

    checkpoint = torch.load(weights_path, map_location='cpu')
    checkpoint_model = checkpoint['model']

    # Count parameters after loading the checkpoint
    count_parameters(model, message="After loading checkpoint")

    # Initialize the extract_embed with the base model and new classifier
    model = classifier_embeddings(model, feat_space, model_name)
    # Load updated checkpoint into the model
    model.load_state_dict(checkpoint_model, strict=False)

    # Count parameters of the custom model
    count_parameters(model, message="Custom model parameters")

    return model

def initialize_model(model_name, feat_space, MODEL_CONSTRUCTORS):
    if model_name in MODEL_CONSTRUCTORS:
        model_constructor = MODEL_CONSTRUCTORS[model_name]
        if model_name == "vit_h_14":
            from torchvision.models import vit_h_14, ViT_H_14_Weights
            weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.DEFAULT
            model = vit_h_14(weights=weights)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            preprocess = weights.transforms()
            data_config = None
            transforms = None
        elif model_name == "regnet_y_128gf":
            from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights
            weights = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1
            model = regnet_y_128gf(weights=weights)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            preprocess = weights.transforms()
            data_config = None
            transforms = None
        elif model_name == "mobilenet_v3_large":
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            model = mobilenet_v3_large(weights=weights)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            preprocess = weights.transforms()
            data_config = None
            transforms = None
        elif model_name in ("mobilenetv4_r448", "eva02_large_patch14_448_embeddings_imageNet"):
            model = model_constructor
            preprocess=None
            data_config = timm.data.resolve_model_data_config(model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            model = classifier_embeddings(base_model=model, feat_space=feat_space, model_name=model_name)
        elif model_name == "mobilenetv4_r448_trained":
            # Pre-trained model with 5 classes
            weights_path = '/home/sebastian/codes/QuantumVE/q_Net/pretrain/mobilenetv4_r448/checkpoint-99.pth'
            model = load_and_initialize_model(model_name, weights_path, 5)
            # import pdb;pdb.set_trace()
            preprocess=None
            data_config = timm.data.resolve_model_data_config(model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            model = classifier_embeddings(base_model=model.base_model, feat_space=feat_space, model_name=model_name)
        else:
            model = model_constructor(pretrained=True, progress=True)
            model.classifier[1].in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, out_features=feat_space)
            preprocess = None
            data_config = None
            transforms = None
        return model, preprocess, transforms, data_config
    else:
        print("Model not available")
        return None

def extract_embeddings(model, data_loader, save_path, device, preprocess=None,data_config=None, transforms=None):
    embeddings_list = []
    targets_list = []
    total_batches = len(data_loader)
    with torch.no_grad(), tqdm(total=total_batches) as pbar:
        model.eval()  # Set the model to evaluation mode
        model.to(device)
        for images, targets in data_loader:
            if preprocess:
                images = preprocess(images).squeeze()
                images = images.to(device)
                embeddings = model(images)
            if transforms: # for timm models
                images = images.to(device)
                embeddings = model(transforms(images))# output is (batch_size, num_features) shaped tensor

            embeddings_list.append(embeddings.cpu().detach().numpy())  # Move to CPU and convert to NumPy
            targets_list.append(targets.numpy())  # Convert targets to NumPy
            pbar.update(1)

    # Concatenate embeddings and targets from all batches
    embeddings = np.concatenate(embeddings_list).squeeze()
    targets = np.concatenate(targets_list)
    num_embeddings = embeddings.shape[1]
    column_names = [f"feat_{i}" for i in range(num_embeddings)]
    column_names.append("label")

    embeddings_with_targets = np.hstack((embeddings, np.expand_dims(targets, axis=1)))

    # Create a DataFrame with column names
    df = pd.DataFrame(embeddings_with_targets, columns=column_names)

    df.to_csv(save_path, index=False)

"""### Extract embeddings"""

for embedding_size in embedding_sizes:
    for batch_size in batch_sizes:
        # Create a unique output directory for each configuration
        experiment_name = f"{model_name}/{model_name}_{embedding_size}_bs{batch_size}"
        os.makedirs(experiment_name, exist_ok=True)

        # Generate command-line arguments for this combination
        args = create_args(batch_size, model_name, embedding_size, model_name, data_path, device)

        print(f"\n{model_name}_{embedding_size}_bs{batch_size}".center(60,"-"))

        print("PARAMETERS\n{}".format(args).replace(', ', ',\n'))

        model, preprocess, transforms, data_config = initialize_model(args.model, args.embedding_size, MODEL_CONSTRUCTORS)
        print(data_config, transforms)
        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)

        os.makedirs(model_name, exist_ok=True)
        device = torch.device(args.device)

        # set seeds
        misc.init_distributed_mode(args)
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        if True:  # args.distributed:
                num_tasks = misc.get_world_size()
                global_rank = misc.get_rank()
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
                print("Sampler_train = %s" % str(sampler_train))
                if args.dist_eval:
                    if len(dataset_val) % num_tasks != 0:
                        print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                              'This will slightly alter validation results as extra duplicate entries are added to achieve '
                              'equal num of samples per-process.')
                    sampler_val = torch.utils.data.DistributedSampler(
                        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
                else:
                    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if global_rank == 0 and args.log_dir is not None and not args.eval:
                os.makedirs(args.log_dir, exist_ok=True)
                log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
                log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
        )
        # Extract embeddings for training data
        extract_embeddings(model, data_loader_train, f'{experiment_name}/train_embeddings.csv', device, preprocess=preprocess, transforms=transforms, data_config=data_config)

        # Extract embeddings for validation data
        extract_embeddings(model, data_loader_val, f'{experiment_name}/val_embeddings.csv', device, preprocess=preprocess,transforms=transforms, data_config=data_config)

        print(f"Completed embeddings extraction for embedding_size={embedding_size} and batch_size={batch_size}")

print("All configurations processed.")