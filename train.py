"""
Fine-tune QuantVGG11Patch on HAM10000 skin cancer dataset.

Expects data downloaded from Kaggle (kmader/skin-cancer-mnist-ham10000):
  data/
    HAM10000_metadata.csv
    HAM10000_images_part_1/
    HAM10000_images_part_2/

Usage:
  python train.py --data-dir ./data --epochs 20 --batch-size 4 --lr 1e-4
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from model import QuantVGG11Patch, PatchAggregator, load_pretrained_weights, NUM_CLASSES


# --- label mapping (alphabetical) ---
DX_TO_IDX = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}
IDX_TO_DX = {v: k for k, v in DX_TO_IDX.items()}


# --- dataset ---
class HAM10000Dataset(Dataset):
    """
    Loads HAM10000 images, center-crops to 224x224, returns 49 patches of 32x32.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


def build_datasets(data_dir):
    """
    Read metadata CSV, locate images, split by lesion_id to prevent leakage.
    Returns train_dataset, val_dataset, class_weights tensor.
    """
    data_dir = Path(data_dir)
    meta = pd.read_csv(data_dir / "HAM10000_metadata.csv")

    # find each image file -- could be in part_1 or part_2
    image_dirs = [
        data_dir / "HAM10000_images_part_1",
        data_dir / "HAM10000_images_part_2",
    ]

    def find_image(image_id):
        for d in image_dirs:
            p = d / f"{image_id}.jpg"
            if p.exists():
                return str(p)
        return None

    meta["path"] = meta["image_id"].apply(find_image)
    meta = meta.dropna(subset=["path"])  # drop if image not found
    meta["label"] = meta["dx"].map(DX_TO_IDX)

    print(f"found {len(meta)} images")
    print(f"class distribution:\n{meta['dx'].value_counts().to_string()}")

    # split by lesion_id so same lesion stays in same split (prevents leakage)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(meta, meta["label"], groups=meta["lesion_id"]))

    train_meta = meta.iloc[train_idx].reset_index(drop=True)
    val_meta = meta.iloc[val_idx].reset_index(drop=True)

    print(f"train: {len(train_meta)}, val: {len(val_meta)}")

    # compute class weights from training set (inverse frequency)
    class_counts = train_meta["label"].value_counts().sort_index()
    total = len(train_meta)
    weights = []
    for i in range(NUM_CLASSES):
        count = class_counts.get(i, 1)
        weights.append(total / (NUM_CLASSES * count))
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"class weights: {class_weights.tolist()}")

    # imagenet normalization
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = HAM10000Dataset(train_meta["path"].tolist(), train_meta["label"].tolist(), transform)
    val_ds = HAM10000Dataset(val_meta["path"].tolist(), val_meta["label"].tolist(), transform)

    return train_ds, val_ds, class_weights


# --- training ---
def pick_device():
    """apple metal if available, else cpu"""
    if torch.backends.mps.is_available():
        print("using Apple Metal (MPS)")
        return torch.device("mps")
    print("MPS not available, using CPU")
    return torch.device("cpu")


def train_one_epoch(model, aggregator, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)  # [B, 3, 224, 224]
        labels = labels.to(device)  # [B]

        optimizer.zero_grad()

        # aggregator splits into patches, runs model, takes max per class
        logits = aggregator(images)  # [B, 7]

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)}, loss: {loss.item():.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, aggregator, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = [0] * NUM_CLASSES
    per_class_total = [0] * NUM_CLASSES

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = aggregator(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i in range(labels.size(0)):
            c = labels[i].item()
            per_class_total[c] += 1
            if preds[i].item() == c:
                per_class_correct[c] += 1

    avg_loss = total_loss / total
    accuracy = correct / total

    print(f"  per-class recall:")
    for i in range(NUM_CLASSES):
        if per_class_total[i] > 0:
            recall = per_class_correct[i] / per_class_total[i]
            print(f"    {IDX_TO_DX[i]:>6s}: {recall:.3f} ({per_class_correct[i]}/{per_class_total[i]})")

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="fine-tune QuantVGG11Patch on HAM10000")
    parser.add_argument("--data-dir", type=str, default="./data", help="path to HAM10000 data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-path", type=str, default="quant_vgg11_patch.pth")
    args = parser.parse_args()

    device = pick_device()

    # build datasets
    train_ds, val_ds, class_weights = build_datasets(args.data_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # build model with pretrained conv weights
    model = QuantVGG11Patch()
    model = load_pretrained_weights(model)
    model = model.to(device)

    # aggregator is a plain python wrapper, not an nn.Module
    aggregator = PatchAggregator(model)

    # weighted loss to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- epoch {epoch + 1}/{args.epochs} ---")

        train_loss, train_acc = train_one_epoch(model, aggregator, train_loader, criterion, optimizer, device)
        print(f"  train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, aggregator, val_loader, criterion, device)
        print(f"  val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  saved best model (val_acc={val_acc:.4f})")

    print(f"\nbest val accuracy: {best_val_acc:.4f}")
    print(f"model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
