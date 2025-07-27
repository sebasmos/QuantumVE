import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
from torchvision.models import efficientnet_b3
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".png") and "_" in f
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        label_str, _ = fname.replace(".png", "").split("_")
        label = int(label_str)
        path = os.path.join(self.image_dir, fname)
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def extract_and_save_embeddings(model, dataloader, device, out_dir, split):
    """
    Extracts embeddings for the given split and saves them as 'train.csv' or 'test.csv'
    inside the model-specific folder.
    """
    all_embs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Extracting {split} embeddings"):
            images = images.to(device)
            embs = model.encode_image(images) if hasattr(model, "encode_image") else model(images)
            all_embs.append(embs.cpu().numpy())
            all_labels.extend(labels.numpy())

    embs_array = np.vstack(all_embs)
    df = pd.DataFrame(embs_array)
    df['label'] = all_labels

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{split}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved {split} embeddings to {save_path} (shape={df.shape})")


def main():
    parser = argparse.ArgumentParser(description="Extract and save image embeddings")
    parser.add_argument(
        "--train_dir", type=str,
        default="../data/fashionmnist_embeddings/distilled_train_images",
        help="Directory with distilled training images"
    )
    parser.add_argument(
        "--test_dir", type=str,
        default="../data/fashionmnist_embeddings/distilled_train_images",
        help="Directory with distilled testing images"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="../data/fashionmnist_embeddings/",
        help="Base directory for saving embeddings"
    )
    parser.add_argument(
        "--batch_size", type=int,
        default=64,
        help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--model_type", type=str,
        choices=["efficientnet", "vit-b/16", "vit-b/32", "vit-l/14", "vit-l/14@336px"],
        default="vit-l/14@336px",
        help="Which model to use for embeddings"
    )
    parser.add_argument(
        "--embedding_size", type=int,
        default=None,
        help="Override output embedding size for EfficientNet"
    )
    args = parser.parse_args()

    # Prepare model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_folder = args.model_type.replace("/", "_").replace("@", "_at_")

    if args.model_type == "efficientnet":
        model = efficientnet_b3(weights="IMAGENET1K_V1")
        in_feat = model.classifier[1].in_features
        out_feat = args.embedding_size if args.embedding_size is not None else in_feat
        model.classifier[1] = nn.Linear(in_feat, out_feat)
        embedding_dim = out_feat
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])
    else:
        import clip
        clip_map = {
            "vit-b/16":   "ViT-B/16",
            "vit-b/32":   "ViT-B/32",
            "vit-l/14":   "ViT-L/14",
            "vit-l/14@336px": "ViT-L/14@336px"
        }
        model, transform = clip.load(clip_map[args.model_type], device=device)
        embedding_dim = model.visual.output_dim

    model.eval().to(device)

    # Create output folder: only one level (model + dim)
    folder_name = f"{model_folder}_{embedding_dim}"
    out_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    # Data loaders
    train_ds = CustomImageDataset(args.train_dir, transform)
    test_ds = CustomImageDataset(args.test_dir, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Extract and save
    extract_and_save_embeddings(model, train_loader, device, out_dir, "train")
    extract_and_save_embeddings(model, test_loader, device, out_dir, "test")

if __name__ == "__main__":
    main()