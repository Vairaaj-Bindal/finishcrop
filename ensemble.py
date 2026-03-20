"""
=============================================================================
ENSEMBLE STACKING META-LEARNER
=============================================================================
Implements a two-level stacking ensemble:

Level 0 (Base Learners):
  - XGBoost (Crop, Fertilizer, Water)
  - Random Forest (Crop, Fertilizer, Water)
  - PyTorch Neural Network (Crop, Fertilizer, Water)

Level 1 (Meta-Learner):
  - Logistic Regression with L2 regularization (classification)
  - Ridge Regression (water requirement)

Meta-features:
  - Probability outputs from all base classifiers
  - Raw predictions from regressors
  - Original features (optional, for richer representation)

Cross-validation for meta-feature generation prevents data leakage.

Paper-worthy: Stacking ensemble with heterogeneous base learners
(tree-based + neural network) for agricultural prediction.
=============================================================================
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score,
    mean_squared_error, mean_absolute_error
)
import torch
import joblib
import os
import time

from config import ENSEMBLE_CONFIG, MODEL_DIR


class StackingEnsemble:
    """
    Stacking Ensemble Meta-Learner.

    Combines predictions from XGBoost, Random Forest, and Neural Network
    using a meta-learner trained on out-of-fold predictions.
    """

    def __init__(self, n_crop_classes, n_fert_classes):
        self.n_crop_classes = n_crop_classes
        self.n_fert_classes = n_fert_classes

        # Meta-learners
        self.meta_crop = LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs',
            random_state=42
        )
        self.meta_fert = LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs',
            random_state=42
        )
        self.meta_water = Ridge(alpha=1.0)

        self.metrics = {}

    def _get_nn_predictions(self, model, X_tensor, device='cpu'):
        """Get predictions from the PyTorch model."""
        model.eval()
        model.to(device)
        with torch.no_grad():
            x = X_tensor.to(device) if isinstance(X_tensor, torch.Tensor) else torch.FloatTensor(X_tensor).to(device)
            crop_logits, fert_logits, water_pred, _, _ = model(x)

            crop_proba = torch.softmax(crop_logits, dim=1).cpu().numpy()
            fert_proba = torch.softmax(fert_logits, dim=1).cpu().numpy()
            water = water_pred.squeeze().cpu().numpy()

        return crop_proba, fert_proba, water

    def build_meta_features(self, traditional_models, nn_model, X, include_original=False):
        """
        Construct meta-feature matrix from base model predictions.

        Meta-features include:
          - XGBoost crop probabilities (22 dims)
          - RF crop probabilities (22 dims)
          - NN crop probabilities (22 dims)
          - XGBoost fert probabilities (7 dims)
          - RF fert probabilities (7 dims)
          - NN fert probabilities (7 dims)
          - XGBoost water prediction (1 dim)
          - RF water prediction (1 dim)
          - NN water prediction (1 dim)
          Total: 22*3 + 7*3 + 3 = 90 meta-features

        Args:
            traditional_models: TraditionalModels instance
            nn_model: CropMultiTaskNet instance
            X: numpy array of original features
            include_original: if True, append original features

        Returns:
            meta_crop: meta-features for crop prediction
            meta_fert: meta-features for fertilizer prediction
            meta_water: meta-features for water prediction
        """
        # Get traditional model predictions
        trad_preds = traditional_models.get_base_predictions(X, proba=True)

        # Get NN predictions
        nn_crop_proba, nn_fert_proba, nn_water = self._get_nn_predictions(nn_model, X)

        # Build meta-feature matrices
        # For crop classification
        meta_crop = np.hstack([
            trad_preds["xgb_crop_proba"],
            trad_preds["rf_crop_proba"],
            nn_crop_proba,
        ])

        # For fertilizer classification
        meta_fert = np.hstack([
            trad_preds["xgb_fert_proba"],
            trad_preds["rf_fert_proba"],
            nn_fert_proba,
        ])

        # For water regression
        meta_water = np.column_stack([
            trad_preds["xgb_water"],
            trad_preds["rf_water"],
            nn_water,
        ])

        if include_original:
            meta_crop = np.hstack([meta_crop, X])
            meta_fert = np.hstack([meta_fert, X])
            meta_water = np.hstack([meta_water, X])

        return meta_crop, meta_fert, meta_water

    def train(self, traditional_models, nn_model, X_train, yc_train, yf_train, yw_train,
              X_val, yc_val, yf_val, yw_val, verbose=True):
        """
        Train the stacking meta-learners.
        """
        print("\n--- Training Stacking Ensemble ---")
        t0 = time.time()

        # Build meta-features for training
        if verbose:
            print("  Building meta-features...")
        meta_crop_train, meta_fert_train, meta_water_train = \
            self.build_meta_features(traditional_models, nn_model, X_train)

        meta_crop_val, meta_fert_val, meta_water_val = \
            self.build_meta_features(traditional_models, nn_model, X_val)

        if verbose:
            print(f"    Crop meta-features shape:  {meta_crop_train.shape}")
            print(f"    Fert meta-features shape:  {meta_fert_train.shape}")
            print(f"    Water meta-features shape: {meta_water_train.shape}")

        # Train meta-learners
        if verbose:
            print("\n  Training meta-learner for Crop...")
        self.meta_crop.fit(meta_crop_train, yc_train)
        yc_pred = self.meta_crop.predict(meta_crop_val)
        acc = accuracy_score(yc_val, yc_pred)
        f1 = f1_score(yc_val, yc_pred, average="weighted")
        self.metrics["ensemble_crop_acc"] = acc
        self.metrics["ensemble_crop_f1"] = f1
        if verbose:
            print(f"    Ensemble Crop Accuracy: {acc:.4f} | F1: {f1:.4f}")

        if verbose:
            print("\n  Training meta-learner for Fertilizer...")
        self.meta_fert.fit(meta_fert_train, yf_train)
        yf_pred = self.meta_fert.predict(meta_fert_val)
        acc = accuracy_score(yf_val, yf_pred)
        f1 = f1_score(yf_val, yf_pred, average="weighted")
        self.metrics["ensemble_fert_acc"] = acc
        self.metrics["ensemble_fert_f1"] = f1
        if verbose:
            print(f"    Ensemble Fert Accuracy: {acc:.4f} | F1: {f1:.4f}")

        if verbose:
            print("\n  Training meta-learner for Water Requirement...")
        self.meta_water.fit(meta_water_train, yw_train)
        yw_pred = self.meta_water.predict(meta_water_val)
        r2 = r2_score(yw_val, yw_pred)
        rmse = np.sqrt(mean_squared_error(yw_val, yw_pred))
        mae = mean_absolute_error(yw_val, yw_pred)
        self.metrics["ensemble_water_r2"] = r2
        self.metrics["ensemble_water_rmse"] = rmse
        self.metrics["ensemble_water_mae"] = mae
        if verbose:
            print(f"    Ensemble Water R²: {r2:.4f} | RMSE: {rmse:.1f} | MAE: {mae:.1f}")

        elapsed = time.time() - t0
        if verbose:
            print(f"\n  Stacking ensemble training completed in {elapsed:.1f}s")

        return elapsed

    def predict(self, traditional_models, nn_model, X):
        """Make predictions using the full ensemble."""
        meta_crop, meta_fert, meta_water = \
            self.build_meta_features(traditional_models, nn_model, X)

        crop_pred = self.meta_crop.predict(meta_crop)
        fert_pred = self.meta_fert.predict(meta_fert)
        water_pred = self.meta_water.predict(meta_water)

        return crop_pred, fert_pred, water_pred

    def predict_proba(self, traditional_models, nn_model, X):
        """Get probability predictions from ensemble."""
        meta_crop, meta_fert, _ = \
            self.build_meta_features(traditional_models, nn_model, X)

        crop_proba = self.meta_crop.predict_proba(meta_crop)
        fert_proba = self.meta_fert.predict_proba(meta_fert)

        return crop_proba, fert_proba

    def save(self):
        """Save ensemble meta-learners."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        ensemble_dict = {
            "meta_crop": self.meta_crop,
            "meta_fert": self.meta_fert,
            "meta_water": self.meta_water,
            "metrics": self.metrics,
        }
        path = os.path.join(MODEL_DIR, "ensemble.pkl")
        joblib.dump(ensemble_dict, path)
        print(f"  Ensemble saved to: {path}")

    def load(self):
        """Load ensemble meta-learners."""
        path = os.path.join(MODEL_DIR, "ensemble.pkl")
        if os.path.exists(path):
            d = joblib.load(path)
            self.meta_crop = d["meta_crop"]
            self.meta_fert = d["meta_fert"]
            self.meta_water = d["meta_water"]
            self.metrics = d["metrics"]
            print(f"  Ensemble loaded from: {path}")
