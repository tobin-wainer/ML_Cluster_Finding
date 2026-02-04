import os
import itertools
import json
import time
import numpy as np
import pandas as pd
from astropy.io import fits

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# Global config / hyperparams for experiment
# ============================================================

# Data paths
F475_DATA_DIR = "/astro/store/gradscratch/tmp/tobinw/PHAT_Cutout_Images/F475W"
F814_DATA_DIR = "/astro/store/gradscratch/tmp/tobinw/PHAT_Cutout_Images/F814W"
LABELS_CSV   = "kam_table_with_test_data_flag.csv"

# Image / model basics
IMG_SIZE     = (301, 301)   # (H, W)
IN_CHANNELS  = 2
KERNEL_SIZE  = 3
BATCH_NORM   = True

# Training controls
MAX_EPOCHS          = 50      # upper limit per config
EARLY_STOP_PATIENCE = 5       # epochs without improvement before stopping
MIN_EPOCHS          = 15       # don't stop before this many epochs
FRAC_DATA           = 0.75    # use only fraction of available data per config
VAL_FRACTION        = 0.2     # fraction of subset used for validation

# Log-stretch
VMAX = 3.0                    # fixed vmax

# System
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = max(1, os.cpu_count() - 1)
PIN_MEMORY  = torch.cuda.is_available()

# Output
OUTPUT_DIR = "grid_search_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Hyperparameter grid
# (start modest; expand once it's working)
# ============================================================

HYPERPARAM_GRID = {
    "dropout":       [0.0, 0.2],
    "lr":            [1e-2, 1e-3, 1e-4],
    "weight_decay":  [0.0, 1e-2, 1e-4],
    "n_layers":      [2, 3, 4, 5],
    "conv_channels": [16, 32],
    "batch_size":    [16, 32, 64, 128, 256],
    "log_a":         [10.0, 20.0, 30, 50],  # log-stretch strength
}

# ============================================================
# Utility
# ============================================================

def format_time(seconds: float) -> str:
    """Return human-readable time string from seconds."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"

# ============================================================
# Dataset
# ============================================================

class FITSDataset(Dataset):
    def __init__(self, csv_file, f475_data_dir, f814_data_dir,
                 target_size=(301, 301), vmax=3.0, log_a=20.0):
        self.df = pd.read_csv(csv_file)
        self.f475_data_dir = f475_data_dir
        self.f814_data_dir = f814_data_dir
        self.target_size = target_size
        self.vmax = vmax
        self.log_a = log_a

        # Filter out test data
        self.df = self.df[self.df["Test_Data_Flag"] == False].reset_index(drop=True)

        # Filter rows where both FITS files exist
        def files_exist(row):
            file1 = os.path.join(f475_data_dir, row['f475_image_string'])
            file2 = os.path.join(f814_data_dir, row['f814_image_string'])
            return os.path.isfile(file1) and os.path.isfile(file2)

        self.df = self.df[self.df.apply(files_exist, axis=1)].reset_index(drop=True)
        print(f"[Dataset] Filtered dataset length (after Test_Data_Flag + existence): {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        band1 = self.load_fits(os.path.join(self.f475_data_dir, row['f475_image_string']))
        band2 = self.load_fits(os.path.join(self.f814_data_dir, row['f814_image_string']))

        stacked = np.stack([band1, band2], axis=0)
        label = np.float32(row['prob'])
        label = np.clip(label, 0, 1)

        stacked = stacked.copy()  # fix negative strides for PyTorch

        return torch.tensor(stacked, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def load_fits(self, filepath):
        with fits.open(filepath, memmap=True) as hdul:
            data = hdul[0].data.astype(np.float32)

        data = np.squeeze(data)

        # Handle non-finite images
        if not np.isfinite(data).any():
            return np.zeros(self.target_size, dtype=np.float32)

        median_val = np.nanmedian(data)
        data = np.nan_to_num(data, nan=median_val)

        H, W = data.shape
        target_H, target_W = self.target_size

        start_H = max((H - target_H) // 2, 0)
        start_W = max((W - target_W) // 2, 0)
        cropped = data[start_H:start_H + target_H, start_W:start_W + target_W]

        pad_H = max(target_H - cropped.shape[0], 0)
        pad_W = max(target_W - cropped.shape[1], 0)
        if pad_H > 0 or pad_W > 0:
            cropped = np.pad(
                cropped,
                ((0, pad_H), (0, pad_W)),
                mode='constant',
                constant_values=median_val
            )

        # Log-stretch
        vmax = self.vmax
        cropped = np.clip(cropped, 0.01, vmax)
        norm = cropped / vmax
        stretched = np.log1p(self.log_a * norm) / np.log1p(self.log_a)

        return stretched.astype(np.float32)

# ============================================================
# Model
# ============================================================

class FlexibleCNN(nn.Module):
    def __init__(self,
                 in_channels,
                 n_layers,
                 conv_channels,
                 kernel_size,
                 img_size,
                 dropout,
                 batch_norm=True):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels, conv_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2)
        )
        layers.append(nn.BatchNorm2d(conv_channels) if batch_norm else nn.Identity())
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout2d(dropout) if dropout > 0 else nn.Identity())

        for _ in range(1, n_layers):
            layers.append(
                nn.Conv2d(conv_channels, conv_channels,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2)
            )
            layers.append(nn.BatchNorm2d(conv_channels) if batch_norm else nn.Identity())
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Dropout2d(dropout) if dropout > 0 else nn.Identity())

        self.conv_model = nn.Sequential(*layers)
        H, W = img_size
        H //= 2 ** n_layers
        W //= 2 ** n_layers
        flattened_size = conv_channels * H * W
        self.fc_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_model(x)
        x = self.fc_model(x)
        return x.squeeze(1)

# ============================================================
# Training routine for a single config
# ============================================================

def train_one_config(config_id, config, dataset):
    """
    Train a model for a single hyperparameter configuration.

    Returns:
        history: list of per-epoch dicts
        summary: dict of final metrics and config
    """
    np.random.seed(42 + config_id)
    torch.manual_seed(42 + config_id)

    # ----- Subsample data (FRAC_DATA) and split train/val -----
    all_indices = np.arange(len(dataset))
    subset_size = max(1, int(FRAC_DATA * len(all_indices)))
    subset_indices = np.random.choice(all_indices, size=subset_size, replace=False)

    train_idx, val_idx = train_test_split(
        subset_indices, test_size=VAL_FRACTION, random_state=42
    )

    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # ----- Model, loss, optimizer -----
    model = FlexibleCNN(
        in_channels=IN_CHANNELS,
        n_layers=config["n_layers"],
        conv_channels=config["conv_channels"],
        kernel_size=KERNEL_SIZE,
        img_size=IMG_SIZE,
        dropout=config["dropout"],
        batch_norm=BATCH_NORM
    ).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    history = []

    # ----- Training loop with early stopping -----
    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_subset)

        # Validation
        model.eval()
        val_loss_total = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item() * images.size(0)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss = val_loss_total / len(val_subset)

        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        r2  = r2_score(all_labels, all_preds)

        print(
            f"[Config {config_id}] Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"MAE: {mae:.4f}"
        )

        # Log this epoch
        history.append({
            "config_id": config_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            **config
        })

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()

        # Early stopping
        if epoch >= MIN_EPOCHS and (epoch - best_epoch) >= EARLY_STOP_PATIENCE:
            print(
                f"[Config {config_id}] Early stopping at epoch {epoch} "
                f"(best epoch {best_epoch}, best val loss {best_val_loss:.4f})"
            )
            break

    # Save best model weights for this config
    model_path = os.path.join(OUTPUT_DIR, f"best_model_config{config_id}.pt")
    if best_state_dict is not None:
        torch.save(best_state_dict, model_path)

    # Final summary for this config (using last epoch metrics)
    last_epoch_metrics = history[-1]
    summary = {
        "config_id": config_id,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": last_epoch_metrics["val_loss"],
        "final_mae": last_epoch_metrics["mae"],
        "final_mse": last_epoch_metrics["mse"],
        "final_r2":  last_epoch_metrics["r2"],
        "model_path": model_path,
        **config
    }

    return history, summary

# ============================================================
# Main grid search
# ============================================================

def main():
    # Fix seeds for reproducibility of splits
    np.random.seed(42)
    torch.manual_seed(42)

    all_histories = []
    all_summaries = []

    # Grid of configs
    config_values = list(HYPERPARAM_GRID.values())
    config_keys   = list(HYPERPARAM_GRID.keys())
    config_iter   = list(itertools.product(*config_values))

    print(f"Total number of configurations: {len(config_iter)}")

    for config_id, values in enumerate(config_iter):
        config = dict(zip(config_keys, values))
        print("\n" + "=" * 60)
        print(f"Starting config {config_id}")
        print(json.dumps(config, indent=2))
        print("=" * 60)

        # Create dataset for this config (log_a affects preprocessing)
        dataset = FITSDataset(
            LABELS_CSV,
            F475_DATA_DIR,
            F814_DATA_DIR,
            target_size=IMG_SIZE,
            vmax=VMAX,
            log_a=config["log_a"]
        )

        start_time = time.time()
        history, summary = train_one_config(config_id, config, dataset)
        elapsed = time.time() - start_time
        summary["train_time_sec"] = elapsed
        summary["train_time_readable"] = format_time(elapsed)

        all_histories.extend(history)
        all_summaries.append(summary)

        print(
            f"[Config {config_id}] Done in {format_time(elapsed)} "
            f"({elapsed:.1f}s). Best val loss: {summary['best_val_loss']:.4f}"
        )

    # Save per-epoch metrics
    history_df = pd.DataFrame(all_histories)
    history_path = os.path.join(OUTPUT_DIR, "gridsearch_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Saved per-epoch history to {history_path}")

    # Save per-config summary
    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(OUTPUT_DIR, "gridsearch_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved per-config summary to {summary_path}")

    # Also save as JSON (optional)
    summary_json_path = os.path.join(OUTPUT_DIR, "gridsearch_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Saved per-config summary JSON to {summary_json_path}")


if __name__ == "__main__":
    main()