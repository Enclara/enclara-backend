"""
Fine-tune QuantVGG11Patch on HAM10000 skin cancer dataset.

Expects data downloaded from HuggingFace (marmal88/skin_cancer) in parquet format:
  data/
    train-00000-of-00005-*.parquet
    train-00001-of-00005-*.parquet
    ...
    validation-00000-of-00002-*.parquet
    validation-00001-of-00002-*.parquet
    test-00000-of-00001-*.parquet

Usage:
  python train.py --data-dir ./data --epochs 20 --batch-size 16 --lr 1e-4
"""

import argparse
from pathlib import Path
import glob

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io

from model import QuantVGG11Patch, PatchAggregator, load_pretrained_weights, NUM_CLASSES


# --- label mapping ---
DX_TO_IDX = {
    "actinic_keratoses": 0,
    "basal_cell_carcinoma": 1,
    "benign_keratosis-like_lesions": 2,
    "dermatofibroma": 3,
    "melanoma": 4,
    "melanocytic_Nevi": 5,
    "vascular_lesions": 6,
}
IDX_TO_DX = {v: k for k, v in DX_TO_IDX.items()}


# --- dataset ---
class HAM10000Dataset(Dataset):
    """
    Loads HAM10000 images, center-crops to 224x224, returns 49 patches of 32x32.
    """

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: list of PIL Images or image bytes
            labels: list of integer labels
            transform: torchvision transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Convert to PIL Image if needed
        if not isinstance(img, Image.Image):
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img)).convert("RGB")
            elif isinstance(img, dict) and 'bytes' in img:
                img = Image.open(io.BytesIO(img['bytes'])).convert("RGB")
            else:
                raise ValueError(f"Unsupported image format: {type(img)}")
        else:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


def build_datasets(data_dir):
    """
    Read parquet files from HuggingFace dataset.
    Returns train_dataset, val_dataset, class_weights tensor.
    """
    data_dir = Path(data_dir)

    # Load all training parquet files
    train_files = sorted(glob.glob(str(data_dir / "train-*.parquet")))
    val_files = sorted(glob.glob(str(data_dir / "validation-*.parquet")))

    if not train_files:
        raise FileNotFoundError(f"No training parquet files found in {data_dir} (expected train-*.parquet)")
    if not val_files:
        raise FileNotFoundError(f"No validation parquet files found in {data_dir} (expected validation-*.parquet)")

    print(f"Found {len(train_files)} training parquet files")
    print(f"Found {len(val_files)} validation parquet files")

    # Read and concatenate all train parquet files
    train_dfs = [pd.read_parquet(f) for f in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True)

    # Read and concatenate all validation parquet files
    val_dfs = [pd.read_parquet(f) for f in val_files]
    val_df = pd.concat(val_dfs, ignore_index=True)

    print(f"Loaded {len(train_df)} training samples")
    print(f"Loaded {len(val_df)} validation samples")

    # Map dx labels to indices
    # HuggingFace parquet may store dx as integers (ClassLabel) or strings
    if train_df["dx"].dtype in ("int64", "int32", "int8"):
        print(f"dx column is integer-encoded (values: {sorted(train_df['dx'].unique())})")
        train_df["label"] = train_df["dx"]
        val_df["label"] = val_df["dx"]
    else:
        print(f"dx column is string-encoded (values: {sorted(train_df['dx'].unique())})")
        train_df["label"] = train_df["dx"].map(DX_TO_IDX)
        val_df["label"] = val_df["dx"].map(DX_TO_IDX)

        # Drop rows with unmapped labels
        train_nan = train_df["label"].isna().sum()
        val_nan = val_df["label"].isna().sum()
        if train_nan > 0 or val_nan > 0:
            print(f"WARNING: dropping {train_nan} train / {val_nan} val rows with unknown dx labels")
            train_df = train_df.dropna(subset=["label"]).reset_index(drop=True)
            val_df = val_df.dropna(subset=["label"]).reset_index(drop=True)

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    print(f"Training class distribution:\n{train_df['dx'].value_counts().to_string()}")

    # Compute class weights from training set (inverse frequency)
    class_counts = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    weights = []
    for i in range(NUM_CLASSES):
        count = class_counts.get(i, 1)
        weights.append(total / (NUM_CLASSES * count))
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"class weights: {class_weights.tolist()}")

    # ImageNet normalization
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract images and labels
    train_images = train_df["image"].tolist()
    train_labels = train_df["label"].tolist()
    val_images = val_df["image"].tolist()
    val_labels = val_df["label"].tolist()

    train_ds = HAM10000Dataset(train_images, train_labels, transform)
    val_ds = HAM10000Dataset(val_images, val_labels, transform)

    return train_ds, val_ds, class_weights


# --- training ---
def pick_device():
    """CUDA if available (for AWS GPU instances), else CPU"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA GPU: {gpu_name}")
        return torch.device("cuda")
    print("CUDA not available, using CPU")
    return torch.device("cpu")


def train_one_epoch(model, aggregator, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)  # [B, 3, 224, 224]
        labels = labels.to(device, non_blocking=True)  # [B]

        optimizer.zero_grad()

        # Use automatic mixed precision if scaler is provided
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = aggregator(images)  # [B, 7]
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                logits = aggregator(images)
                loss = criterion(logits, labels)
        else:
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
    parser.add_argument("--batch-size", type=int, default=16, help="batch size (default: 16 for g4dn.xlarge)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=3, help="dataloader workers (default: 3 for g4dn.xlarge)")
    parser.add_argument("--save-path", type=str, default="quant_vgg11_patch.pth")
    parser.add_argument("--no-amp", action="store_true", help="disable automatic mixed precision")
    args = parser.parse_args()

    device = pick_device()

    # build datasets
    train_ds, val_ds, class_weights = build_datasets(args.data_dir)

    # Optimize DataLoader for GPU training
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),  # faster CPU-to-GPU transfer
        persistent_workers=args.num_workers > 0  # keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0
    )

    # build model with pretrained conv weights
    model = QuantVGG11Patch()
    model = load_pretrained_weights(model)
    model = model.to(device)

    # aggregator is a plain python wrapper, not an nn.Module
    aggregator = PatchAggregator(model)

    # weighted loss to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize GradScaler for automatic mixed precision (AMP)
    scaler = None
    if torch.cuda.is_available() and not args.no_amp:
        scaler = torch.amp.GradScaler("cuda")
        print("Using automatic mixed precision (AMP) for faster training")

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- epoch {epoch + 1}/{args.epochs} ---")

        train_loss, train_acc = train_one_epoch(model, aggregator, train_loader, criterion, optimizer, device, scaler)
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
