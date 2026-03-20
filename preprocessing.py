"""
=============================================================================
DATA PREPROCESSING & FEATURE ENGINEERING PIPELINE
=============================================================================
Handles:
  - Loading and cleaning data (120,000 rows × 26 columns)
  - Feature engineering: nutrient balance score, water stress index, GDD
  - Graceful pass-through for pre-computed CSV columns
  - Label encoding for categorical targets (1,920 crop classes, 15 fert types)
  - One-hot encoding for categorical features
  - RobustScaler feature scaling (handles outliers better than StandardScaler)
  - Stratified train/val/test split
  - PyTorch DataLoader creation
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib
import os
import json

from config import (
    DATASET_PATH, MODEL_DIR, RAW_FEATURES, ENGINEERED_FEATURES,
    CATEGORICAL_FEATURES, TARGET_CROP, TARGET_FERTILIZER, TARGET_WATER,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE, NN_CONFIG
)


class CropDataset(Dataset):
    """PyTorch Dataset for multi-task crop prediction."""

    def __init__(self, X, y_crop, y_fert, y_water):
        self.X = torch.FloatTensor(X)
        self.y_crop = torch.LongTensor(y_crop)
        self.y_fert = torch.LongTensor(y_fert)
        self.y_water = torch.FloatTensor(y_water)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_crop[idx], self.y_fert[idx], self.y_water[idx]


class DataPreprocessor:
    """Complete data preprocessing pipeline."""

    def __init__(self):
        self.crop_encoder = LabelEncoder()
        self.fert_encoder = LabelEncoder()
        self.scaler = None
        self.feature_names = None
        self.n_crop_classes = None
        self.n_fert_classes = None
        self.n_features = None

    def load_data(self):
        """Load the dataset from CSV."""
        print("Loading dataset...")
        df = pd.read_csv(DATASET_PATH)
        print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df

    def _ensure_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute engineered features that are not already in the CSV.
        Pre-computed columns from the generator (vpd, N_P_ratio, et0_daily, etc.)
        are passed through unchanged. Only freshly derived features are computed here.
        """
        eps = 1e-8

        # ── Pass-through check: only compute if column is missing ──────────
        if "N_P_ratio" not in df.columns:
            df["N_P_ratio"] = (df["N"] / (df["P"] + eps)).round(4)
        if "N_K_ratio" not in df.columns:
            df["N_K_ratio"] = (df["N"] / (df["K"] + eps)).round(4)
        if "P_K_ratio" not in df.columns:
            df["P_K_ratio"] = (df["P"] / (df["K"] + eps)).round(4)
        if "total_NPK" not in df.columns:
            df["total_NPK"] = (df["N"] + df["P"] + df["K"]).round(2)
        if "vpd" not in df.columns:
            es = 0.6108 * np.exp(17.27 * df["temperature"] / (df["temperature"] + 237.3))
            df["vpd"] = (es * (1 - df["humidity"] / 100.0)).clip(lower=0).round(4)
        if "heat_stress_index" not in df.columns:
            df["heat_stress_index"] = np.maximum(0, (df["temperature"] - 30) / 10).round(4)
        if "cold_stress_index" not in df.columns:
            df["cold_stress_index"] = np.maximum(0, (15 - df["temperature"]) / 10).round(4)
        if "ph_deviation" not in df.columns:
            df["ph_deviation"] = (df["ph"] - 6.5).abs().round(3)
        if "aridity_index" not in df.columns:
            df["aridity_index"] = (df["rainfall"] / (df["temperature"] + 10 + eps)).round(4)

        # ── Freshly computed features (never in CSV) ───────────────────────
        # Nutrient Balance Score: geometric mean of normalized NPK
        # NBS = (N/N̄ · P/P̄ · K/K̄)^(1/3)  — values near 1.0 indicate balance
        N_norm = df["N"] / (df["N"].mean() + eps)
        P_norm = df["P"] / (df["P"].mean() + eps)
        K_norm = df["K"] / (df["K"].mean() + eps)
        df["nutrient_balance_score"] = (N_norm * P_norm * K_norm).clip(lower=0) ** (1 / 3)

        # Water Stress Index: actual rainfall vs. reference demand
        # WSI = rainfall / (ET0_daily × 30)  — >1 = surplus, <1 = deficit
        if "et0_daily" in df.columns:
            df["water_stress_index"] = (
                df["rainfall"] / (df["et0_daily"] * 30 + eps)
            ).clip(lower=0, upper=10).round(4)
        else:
            df["water_stress_index"] = 1.0  # neutral if ET0 not available

        # Growing Degree Days (GDD) proxy: max(0, T - T_base) × season_length
        # Base temperature T_base = 10°C (common agronomic standard)
        if "growing_days" in df.columns:
            df["growing_degree_days"] = (
                np.maximum(0, df["temperature"] - 10) * df["growing_days"]
            ).round(1)
        else:
            df["growing_degree_days"] = np.maximum(0, df["temperature"] - 10) * 90

        return df

    def prepare_features(self, df: pd.DataFrame):
        """Prepare feature matrix X and target vectors y_*."""
        df = self._ensure_engineered_features(df)

        # Numeric features: raw sensor inputs + engineered
        numeric_features = [f for f in RAW_FEATURES + ENGINEERED_FEATURES if f in df.columns]

        # One-hot encode categorical features
        cat_dfs = []
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                cat_dfs.append(dummies)

        X_numeric = df[numeric_features].values.astype(np.float32)
        if cat_dfs:
            X_cat = pd.concat(cat_dfs, axis=1).astype(np.float32).values
            X = np.hstack([X_numeric, X_cat])
            self.feature_names = numeric_features + list(pd.concat(cat_dfs, axis=1).columns)
        else:
            X = X_numeric
            self.feature_names = numeric_features

        # Encode targets
        y_crop = self.crop_encoder.fit_transform(df[TARGET_CROP])
        y_fert = self.fert_encoder.fit_transform(df[TARGET_FERTILIZER])
        y_water = df[TARGET_WATER].values.astype(np.float32)

        self.n_crop_classes = len(self.crop_encoder.classes_)
        self.n_fert_classes = len(self.fert_encoder.classes_)
        self.n_features = X.shape[1]

        print(f"  Feature matrix shape:   {X.shape}")
        print(f"  Crop classes:           {self.n_crop_classes:,}")
        print(f"  Fertilizer classes:     {self.n_fert_classes}")
        print(f"  Water req range:        [{y_water.min():.0f}, {y_water.max():.0f}] mm")

        return X, y_crop, y_fert, y_water

    def split_data(self, X, y_crop, y_fert, y_water):
        """Stratified split into train/val/test on crop label."""
        X_trainval, X_test, yc_tv, yc_test, yf_tv, yf_test, yw_tv, yw_test = \
            train_test_split(X, y_crop, y_fert, y_water,
                             test_size=TEST_SIZE, random_state=RANDOM_STATE,
                             stratify=y_crop)

        val_ratio = VAL_SIZE / (1 - TEST_SIZE)
        X_train, X_val, yc_train, yc_val, yf_train, yf_val, yw_train, yw_val = \
            train_test_split(X_trainval, yc_tv, yf_tv, yw_tv,
                             test_size=val_ratio, random_state=RANDOM_STATE,
                             stratify=yc_tv)

        print(f"\n  Train: {X_train.shape[0]:,} samples")
        print(f"  Val:   {X_val.shape[0]:,} samples")
        print(f"  Test:  {X_test.shape[0]:,} samples")

        return (X_train, yc_train, yf_train, yw_train,
                X_val, yc_val, yf_val, yw_val,
                X_test, yc_test, yf_test, yw_test)

    def scale_features(self, X_train, X_val, X_test):
        """Scale features using RobustScaler (resistant to outliers)."""
        self.scaler = RobustScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)
        print("  Features scaled with RobustScaler")
        return X_train_s, X_val_s, X_test_s

    def create_dataloaders(self, X_train, yc_train, yf_train, yw_train,
                           X_val, yc_val, yf_val, yw_val,
                           X_test, yc_test, yf_test, yw_test):
        """Create PyTorch DataLoaders."""
        bs = NN_CONFIG["batch_size"]
        train_ds = CropDataset(X_train, yc_train, yf_train, yw_train)
        val_ds   = CropDataset(X_val,   yc_val,   yf_val,   yw_val)
        test_ds  = CropDataset(X_test,  yc_test,  yf_test,  yw_test)

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

        print(f"  DataLoaders ready (batch_size={bs})")
        print(f"    Train: {len(train_loader)} batches")
        print(f"    Val:   {len(val_loader)} batches")
        print(f"    Test:  {len(test_loader)} batches")
        return train_loader, val_loader, test_loader

    def save_preprocessor(self):
        """Persist all preprocessing artifacts for inference."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        artifacts = {
            "crop_encoder":    self.crop_encoder,
            "fert_encoder":    self.fert_encoder,
            "scaler":          self.scaler,
            "feature_names":   self.feature_names,
            "n_crop_classes":  self.n_crop_classes,
            "n_fert_classes":  self.n_fert_classes,
            "n_features":      self.n_features,
        }
        path = os.path.join(MODEL_DIR, "preprocessor.pkl")
        joblib.dump(artifacts, path)
        print(f"  Preprocessor saved → {path}")

        # Also save feature names as JSON for interpretability
        meta_path = os.path.join(MODEL_DIR, "feature_names.json")
        with open(meta_path, "w") as f:
            json.dump({"feature_names": self.feature_names,
                       "n_features": self.n_features,
                       "crop_classes": list(self.crop_encoder.classes_),
                       "fert_classes": list(self.fert_encoder.classes_)}, f, indent=2)
        print(f"  Feature metadata saved → {meta_path}")

    def run_full_pipeline(self):
        """Execute the complete preprocessing pipeline end-to-end."""
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING PIPELINE")
        print("=" * 60)

        df = self.load_data()

        print("\nPreparing features...")
        X, y_crop, y_fert, y_water = self.prepare_features(df)

        print("\nSplitting data (stratified on crop label)...")
        splits = self.split_data(X, y_crop, y_fert, y_water)
        X_train, yc_train, yf_train, yw_train = splits[0:4]
        X_val,   yc_val,   yf_val,   yw_val   = splits[4:8]
        X_test,  yc_test,  yf_test,  yw_test  = splits[8:12]

        print("\nScaling features...")
        X_train_s, X_val_s, X_test_s = self.scale_features(X_train, X_val, X_test)

        print("\nCreating DataLoaders...")
        train_loader, val_loader, test_loader = self.create_dataloaders(
            X_train_s, yc_train, yf_train, yw_train,
            X_val_s,   yc_val,   yf_val,   yw_val,
            X_test_s,  yc_test,  yf_test,  yw_test)

        print("\nSaving preprocessing artifacts...")
        self.save_preprocessor()

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)

        return {
            "train_loader": train_loader,
            "val_loader":   val_loader,
            "test_loader":  test_loader,
            "X_train": X_train_s, "X_val": X_val_s, "X_test": X_test_s,
            "yc_train": yc_train, "yc_val": yc_val,  "yc_test": yc_test,
            "yf_train": yf_train, "yf_val": yf_val,  "yf_test": yf_test,
            "yw_train": yw_train, "yw_val": yw_val,  "yw_test": yw_test,
            "n_features":     self.n_features,
            "n_crop_classes": self.n_crop_classes,
            "n_fert_classes": self.n_fert_classes,
            "feature_names":  self.feature_names,
            "crop_encoder":   self.crop_encoder,
            "fert_encoder":   self.fert_encoder,
        }


if __name__ == "__main__":
    p = DataPreprocessor()
    data = p.run_full_pipeline()
    print(f"\nReady: {data['n_features']} features · "
          f"{data['n_crop_classes']:,} crop classes · "
          f"{data['n_fert_classes']} fertilizer classes")
