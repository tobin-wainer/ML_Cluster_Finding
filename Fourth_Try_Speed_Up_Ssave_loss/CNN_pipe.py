import os
import numpy as np
import pandas as pd
from astropy.io import fits
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import multiprocessing
import pickle

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
K_FOLDS = 3

IMG_SIZE = (301, 301)         # height, width of images
IN_CHANNELS = 2               # two bands
N_LAYERS = 3                  # number of conv layers
CONV_CHANNELS = 32            # channels in conv layers
KERNEL_SIZE = 3
DROPOUT = 0.2
BATCH_NORM = True

# Data handling
NUM_WORKERS = max(1, os.cpu_count() - 1)
PIN_MEMORY = torch.cuda.is_available()  # future-proof: True only if you ever use CUDA

# Log-stretch / scaling
VMAX = 3.0
LOG_A = 20.0   # strength of log stretch

F475_DATA_DIR = "/astro/store/gradscratch/tmp/tobinw/PHAT_Cutout_Images/F475W"
F814_DATA_DIR = "/astro/store/gradscratch/tmp/tobinw/PHAT_Cutout_Images/F814W"
LABELS_CSV = "kam_table_with_test_data_flag.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FITSDataset(Dataset):
    def __init__(self, csv_file, f475_data_dir, f814_data_dir, target_size=(301, 301)):
        self.df = pd.read_csv(csv_file)
        self.f475_data_dir = f475_data_dir
        self.f814_data_dir = f814_data_dir
        self.target_size = target_size

        # Filter out test data
        self.df = self.df[self.df["Test_Data_Flag"] == False].reset_index(drop=True)

        # Only check existence; no FITS opening here
        def files_exist(row):
            file1 = os.path.join(f475_data_dir, row['f475_image_string'])
            file2 = os.path.join(f814_data_dir, row['f814_image_string'])
            return os.path.isfile(file1) and os.path.isfile(file2)

        self.df = self.df[self.df.apply(files_exist, axis=1)].reset_index(drop=True)
        print(f"Filtered dataset length (after Test_Data_Flag + existence): {len(self.df)}")

        # Expand rows for augmentation
        aug_types = ['none', 'rot90', 'rot180', 'rot270', 'flipud', 'fliplr']
        expanded_rows = []
        for _, row in self.df.iterrows():
            prob = np.float32(row['prob'])
            if prob > 0.1:
                for aug in aug_types:
                    new_row = row.copy()
                    new_row['aug'] = aug
                    expanded_rows.append(new_row)
            else:
                new_row = row.copy()
                new_row['aug'] = 'none'
                expanded_rows.append(new_row)
        self.df = pd.DataFrame(expanded_rows)
        print(f"Filtered and augmented dataset length: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        band1 = self.load_fits(os.path.join(self.f475_data_dir, row['f475_image_string']))
        band2 = self.load_fits(os.path.join(self.f814_data_dir, row['f814_image_string']))

        stacked = np.stack([band1, band2], axis=0)
        label = np.float32(row['prob'])
        label = np.clip(label, 0, 1)

        # Augment
        aug = row['aug']
        if aug == 'rot90':
            stacked = np.rot90(stacked, k=1, axes=(1, 2))
        elif aug == 'rot180':
            stacked = np.rot90(stacked, k=2, axes=(1, 2))
        elif aug == 'rot270':
            stacked = np.rot90(stacked, k=3, axes=(1, 2))
        elif aug == 'flipud':
            stacked = np.flip(stacked, axis=1)
        elif aug == 'fliplr':
            stacked = np.flip(stacked, axis=2)

        stacked = stacked.copy()  # fix negative strides

        return torch.tensor(stacked, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def load_fits(self, filepath):
        with fits.open(filepath, memmap=True) as hdul:
            data = hdul[0].data.astype(np.float32)

        data = np.squeeze(data)

        # Handle non-finite images here
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

        # -------------------------------
        # LOG-STRETCH TRANSFORMATION
        # -------------------------------
        cropped = np.clip(cropped, 0.01, VMAX)

        # Normalize to [0,1]
        norm = cropped / VMAX

        stretched = np.log1p(LOG_A * norm) / np.log1p(LOG_A)

        return stretched.astype(np.float32)


class FlexibleCNN(nn.Module):
    def __init__(self,
                 in_channels=IN_CHANNELS,
                 n_layers=N_LAYERS,
                 conv_channels=CONV_CHANNELS,
                 kernel_size=KERNEL_SIZE,
                 img_size=IMG_SIZE,
                 dropout=DROPOUT,
                 batch_norm=BATCH_NORM):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2))
        layers.append(nn.BatchNorm2d(conv_channels) if batch_norm else nn.Identity())
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout2d(dropout) if dropout > 0 else nn.Identity())
        for i in range(1, n_layers):
            layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2))
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
        self.loss = nn.BCELoss()

    def forward(self, x):
        x = self.conv_model(x)
        x = self.fc_model(x)
        return x.squeeze(1)

    def configure_optimizers(self, learning_rate=LR, weight_decay=WEIGHT_DECAY):
        return optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)


def run_fold(fold, train_idx, val_idx, dataset):
    # Use already-created dataset
    print(f"\nFold {fold+1}")
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # DataLoaders use global BATCH_SIZE / NUM_WORKERS
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Model uses global architecture hyperparameters
    model = FlexibleCNN(
        in_channels=IN_CHANNELS,
        n_layers=N_LAYERS,
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        img_size=IMG_SIZE,
        dropout=DROPOUT,
        batch_norm=BATCH_NORM
    ).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = model.configure_optimizers(learning_rate=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    best_epoch = -1

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # ---- Train ----
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

        # ---- Val ----
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

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Fold {fold+1}, Epoch {epoch+1}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            os.makedirs("fold_models", exist_ok=True)
            torch.save(model.state_dict(), f"fold_models/best_model_fold{fold+1}.pt")
            print(f"Saved best model for fold {fold+1} at epoch {best_epoch}")

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    print(f"Fold {fold+1} Metrics:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    return {
        "fold": fold+1,
        "val_idx": val_idx,
        "all_labels": all_labels,
        "all_preds": all_preds,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses
    }


def main():
    # Create the dataset ONCE and reuse it across folds
    dataset = FITSDataset(LABELS_CSV, F475_DATA_DIR, F814_DATA_DIR, target_size=IMG_SIZE)

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        results.append(run_fold(fold, train_idx, val_idx, dataset))

    fold_results = {
        res["fold"]: {
            "val_idx": res["val_idx"],
            "all_labels": res["all_labels"],
            "all_preds": res["all_preds"]
        }
        for res in results
    }

    best_val_losses = [res["best_val_loss"] for res in results]

    with open("fold_val_results.pkl", "wb") as f:
        pickle.dump(fold_results, f)

    avg_val_loss = sum(best_val_losses) / len(best_val_losses)
    print(f"\nAverage best validation loss across all folds: {avg_val_loss:.4f}")

    # Save per-epoch loss curves
    all_logs = []
    for res in results:
        fold = res["fold"]
        for epoch, (tr, va) in enumerate(zip(res["train_losses"], res["val_losses"]), start=1):
            all_logs.append({
                "fold": fold,
                "epoch": epoch,
                "train_loss": tr,
                "val_loss": va
            })

    os.makedirs("fold_models", exist_ok=True)
    loss_df = pd.DataFrame(all_logs)
    loss_df.to_csv("fold_models/training_curves_per_fold.csv", index=False)
    print("Saved training curves to fold_models/training_curves_per_fold.csv")


if __name__ == "__main__":
    print("Staring!")
    main()