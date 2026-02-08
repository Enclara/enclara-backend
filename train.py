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
import time

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
    num_batches = len(loader)
    log_interval = max(1, num_batches // 10)  # log ~10 times per epoch
    epoch_start = time.time()

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

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            running_loss = total_loss / total
            running_acc = correct / total
            elapsed = time.time() - epoch_start
            samples_per_sec = total / elapsed
            print(f"  [{batch_idx + 1:>4d}/{num_batches}] "
                  f"loss: {loss.item():.4f} (avg: {running_loss:.4f}) | "
                  f"acc: {running_acc:.4f} | "
                  f"{samples_per_sec:.1f} samples/sec")

    avg_loss = total_loss / total
    accuracy = correct / total
    epoch_time = time.time() - epoch_start
    print(f"  TRAIN complete: {epoch_time:.1f}s | loss: {avg_loss:.4f} | acc: {accuracy:.4f}")
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, aggregator, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = [0] * NUM_CLASSES
    per_class_total = [0] * NUM_CLASSES
    eval_start = time.time()
    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
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

        if (batch_idx + 1) == num_batches:
            elapsed = time.time() - eval_start
            print(f"  VAL complete: {elapsed:.1f}s | evaluated {total} samples")

    avg_loss = total_loss / total
    accuracy = correct / total

    print(f"  VAL loss: {avg_loss:.4f} | acc: {accuracy:.4f} ({correct}/{total})")
    print(f"  Per-class breakdown:")
    print(f"    {'Class':<35s} {'Recall':>8s}  {'Correct':>8s}  {'Total':>8s}")
    print(f"    {'-'*35} {'-'*8}  {'-'*8}  {'-'*8}")
    for i in range(NUM_CLASSES):
        if per_class_total[i] > 0:
            recall = per_class_correct[i] / per_class_total[i]
            print(f"    {IDX_TO_DX[i]:<35s} {recall:>8.3f}  {per_class_correct[i]:>8d}  {per_class_total[i]:>8d}")

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

    print("=" * 70)
    print("  QuantVGG11Patch Fine-Tuning on Skin Cancer Dataset")
    print("=" * 70)
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Num workers:  {args.num_workers}")
    print(f"  Save path:    {args.save_path}")
    print(f"  AMP:          {'disabled' if args.no_amp else 'enabled'}")
    print("=" * 70)

    device = pick_device()

    # build datasets
    print("\n--- Loading datasets ---")
    load_start = time.time()
    train_ds, val_ds, class_weights = build_datasets(args.data_dir)
    print(f"Datasets loaded in {time.time() - load_start:.1f}s")

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

    print(f"\n{'=' * 70}")
    print(f"  Starting training: {len(train_ds)} train / {len(val_ds)} val samples")
    print(f"  {len(train_loader)} train batches / {len(val_loader)} val batches per epoch")
    print(f"{'=' * 70}")

    training_start = time.time()
    best_val_acc = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\n{'=' * 70}")
        print(f"  EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'=' * 70}")

        train_loss, train_acc = train_one_epoch(model, aggregator, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, aggregator, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - training_start
        epochs_left = args.epochs - (epoch + 1)
        avg_epoch_time = total_elapsed / (epoch + 1)
        eta = avg_epoch_time * epochs_left

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.save_path)
            improved = " ** NEW BEST **"

        print(f"\n  Epoch {epoch + 1} summary:")
        print(f"    Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"    Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}{improved}")
        print(f"    Best so far: {best_val_acc:.4f} (epoch {best_epoch})")
        print(f"    Epoch time: {epoch_time:.1f}s | Elapsed: {total_elapsed:.0f}s | ETA: {eta:.0f}s")

    total_time = time.time() - training_start
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time:       {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  Model saved to:   {args.save_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
