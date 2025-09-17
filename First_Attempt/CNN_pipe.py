import os
import numpy as np
import pandas as pd
from astropy.io import fits
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, brier_score_loss, accuracy_score, recall_score, precision_score
import multiprocessing
import pickle

# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
K_FOLDS = 5

IMG_SIZE = (301, 301)         # height, width of images
IN_CHANNELS = 2               # two bands
N_LAYERS = 3                  # number of conv layers
CONV_CHANNELS = 32            # channels in conv layers
KERNEL_SIZE = 3
DROPOUT = 0.2
BATCH_NORM = True

F475_DATA_DIR = "/astro/store/gradscratch/tmp/tobinw/PHAT_Cutout_Images/F475W"
F814_DATA_DIR = "/astro/store/gradscratch/tmp/tobinw/PHAT_Cutout_Images/F814W"
LABELS_CSV = "kam_table_with_test_data_flag.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FITSDataset(Dataset):
    def __init__(self, csv_file, f475_data_dir, f814_data_dir, target_size=(301,301)):
        self.df = pd.read_csv(csv_file)
        self.f475_data_dir = f475_data_dir
        self.f814_data_dir = f814_data_dir
        self.target_size = target_size

        def files_exist_and_not_all_nan(row):
            file1 = os.path.join(f475_data_dir, row['f475_image_string'])
            file2 = os.path.join(f814_data_dir, row['f814_image_string'])
            if not (os.path.isfile(file1) and os.path.isfile(file2)):
                return False
            # Check if either image is all NaN
            try:
                with fits.open(file1) as hdul1:
                    data1 = hdul1[0].data
                with fits.open(file2) as hdul2:
                    data2 = hdul2[0].data
                if np.all(np.isnan(data1)) or np.all(np.isnan(data2)):
                    return False
            except Exception as e:
                print(f"Error reading {file1} or {file2}: {e}")
                return False
            return True

        self.df = self.df[self.df.apply(files_exist_and_not_all_nan, axis=1)].reset_index(drop=True)
        self.df = self.df[self.df["Test_Data_Flag"] == False].reset_index(drop=True)
        print(f"Filtered dataset length: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        band1 = self.load_fits(os.path.join(self.f475_data_dir, row['f475_image_string']))
        band2 = self.load_fits(os.path.join(self.f814_data_dir, row['f814_image_string']))
        stacked = np.stack([band1, band2], axis=0)
        label = np.float32(row['prob'])
        label = np.clip(label, 0, 1)  # Ensure label is in [0, 1]
        return torch.tensor(stacked, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def load_fits(self, filepath):
        with fits.open(filepath) as hdul:
            data = hdul[0].data.astype(np.float32)
        data = np.squeeze(data)
        if np.all(np.isnan(data)):
            raise ValueError(f"All-NaN image encountered: {filepath}")
        median_val = np.nanmedian(data)
        data = np.nan_to_num(data, nan=median_val)
        H, W = data.shape
        target_H, target_W = self.target_size
        start_H = max((H - target_H) // 2, 0)
        start_W = max((W - target_W) // 2, 0)
        cropped = data[start_H:start_H+target_H, start_W:start_W+target_W]
        pad_H = max(target_H - cropped.shape[0], 0)
        pad_W = max(target_W - cropped.shape[1], 0)
        if pad_H > 0 or pad_W > 0:
            cropped = np.pad(cropped, ((0, pad_H), (0, pad_W)), mode='constant', constant_values=median_val)
        return cropped

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
        layers.append(nn.Conv2d(in_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2))
        layers.append(nn.BatchNorm2d(conv_channels) if batch_norm else nn.Identity())
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout2d(dropout) if dropout > 0 else nn.Identity())
        for i in range(1, n_layers):
            layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2))
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

def run_fold(args):
    fold, train_idx, val_idx = args
    dataset = FITSDataset(LABELS_CSV, F475_DATA_DIR, F814_DATA_DIR)
    print(f"\nFold {fold+1}")
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    model = FlexibleCNN(in_channels=IN_CHANNELS, n_layers=N_LAYERS, conv_channels=CONV_CHANNELS, 
                        kernel_size=KERNEL_SIZE, img_size=IMG_SIZE, dropout=DROPOUT, batch_norm=BATCH_NORM)
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = model.configure_optimizers(learning_rate=LR, weight_decay=WEIGHT_DECAY)
    best_val_loss = float('inf')
    all_preds = []
    all_labels = []
    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0
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
        val_loss_total = 0
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
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"fold_models/best_model_fold{fold+1}.pt")
            print(f"Saved best model for fold {fold+1}")
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    print(f"Fold {fold+1} Metrics:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return {
        "fold": fold+1,
        "val_idx": val_idx,
        "all_labels": all_labels,
        "all_preds": all_preds,
        "best_val_loss": best_val_loss
    }

def main():
    dataset = FITSDataset(LABELS_CSV, F475_DATA_DIR, F814_DATA_DIR)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_args = [(fold, train_idx, val_idx) for fold, (train_idx, val_idx) in enumerate(kf.split(dataset))]
    with multiprocessing.Pool(processes=min(K_FOLDS, 4)) as pool:
        results = pool.map(run_fold, fold_args)
    fold_results = {res["fold"]: {"val_idx": res["val_idx"], "all_labels": res["all_labels"], "all_preds": res["all_preds"]} for res in results}
    best_val_losses = [res["best_val_loss"] for res in results]
    with open("fold_val_results.pkl", "wb") as f:
        pickle.dump(fold_results, f)
    avg_val_loss = sum(best_val_losses) / len(best_val_losses)
    print(f"\nAverage best validation loss across all folds: {avg_val_loss:.4f}")

if __name__ == "__main__":
    main()