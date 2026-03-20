"""
=============================================================================
TRADITIONAL ML MODELS: XGBoost + Random Forest
=============================================================================
Separate models for each task:
  - Crop classification (XGBoost + Random Forest)
  - Fertilizer classification (XGBoost + Random Forest)
  - Water requirement regression (XGBoost + Random Forest)

These serve as both:
  1. Strong baselines for comparison in the research paper
  2. Base learners for the ensemble stacking meta-learner
=============================================================================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    r2_score, mean_squared_error, mean_absolute_error
)
import xgboost as xgb
import joblib
import os
import time

from config import XGB_CONFIG, RF_CONFIG, MODEL_DIR


class TraditionalModels:
    """Wrapper for XGBoost and Random Forest models for all three tasks."""

    def __init__(self):
        # XGBoost models
        self.xgb_crop = None
        self.xgb_fert = None
        self.xgb_water = None

        # Random Forest models
        self.rf_crop = None
        self.rf_fert = None
        self.rf_water = None

        # Training metrics
        self.metrics = {}

    def _build_xgb_models(self, n_crop_classes, n_fert_classes):
        """Initialize XGBoost models."""
        crop_params = XGB_CONFIG["crop"].copy()
        crop_params["num_class"] = n_crop_classes
        early_stop_crop = crop_params.pop("early_stopping_rounds")

        fert_params = XGB_CONFIG["fertilizer"].copy()
        fert_params["num_class"] = n_fert_classes
        early_stop_fert = fert_params.pop("early_stopping_rounds")

        water_params = XGB_CONFIG["water"].copy()
        early_stop_water = water_params.pop("early_stopping_rounds")

        self.xgb_crop = xgb.XGBClassifier(**crop_params)
        self.xgb_crop._early_stopping_rounds = early_stop_crop

        self.xgb_fert = xgb.XGBClassifier(**fert_params)
        self.xgb_fert._early_stopping_rounds = early_stop_fert

        self.xgb_water = xgb.XGBRegressor(**water_params)
        self.xgb_water._early_stopping_rounds = early_stop_water

        print("  XGBoost models initialized")

    def _build_rf_models(self):
        """Initialize Random Forest models."""
        self.rf_crop = RandomForestClassifier(**RF_CONFIG["crop"])
        self.rf_fert = RandomForestClassifier(**RF_CONFIG["fertilizer"])
        self.rf_water = RandomForestRegressor(**RF_CONFIG["water"])
        print("  Random Forest models initialized")

    def train_xgboost(self, X_train, yc_train, yf_train, yw_train,
                      X_val, yc_val, yf_val, yw_val, verbose=True):
        """Train all XGBoost models."""
        print("\n--- Training XGBoost Models ---")
        timers = {}

        # Crop classification
        if verbose:
            print("\n  [XGB] Training Crop Classifier...")
        t0 = time.time()
        self.xgb_crop.fit(
            X_train, yc_train,
            eval_set=[(X_val, yc_val)],
            verbose=False
        )
        timers["xgb_crop"] = time.time() - t0
        yc_pred = self.xgb_crop.predict(X_val)
        acc = accuracy_score(yc_val, yc_pred)
        f1 = f1_score(yc_val, yc_pred, average="weighted")
        self.metrics["xgb_crop_acc"] = acc
        self.metrics["xgb_crop_f1"] = f1
        if verbose:
            print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {timers['xgb_crop']:.1f}s")

        # Fertilizer classification
        if verbose:
            print("\n  [XGB] Training Fertilizer Classifier...")
        t0 = time.time()
        self.xgb_fert.fit(
            X_train, yf_train,
            eval_set=[(X_val, yf_val)],
            verbose=False
        )
        timers["xgb_fert"] = time.time() - t0
        yf_pred = self.xgb_fert.predict(X_val)
        acc = accuracy_score(yf_val, yf_pred)
        f1 = f1_score(yf_val, yf_pred, average="weighted")
        self.metrics["xgb_fert_acc"] = acc
        self.metrics["xgb_fert_f1"] = f1
        if verbose:
            print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {timers['xgb_fert']:.1f}s")

        # Water regression
        if verbose:
            print("\n  [XGB] Training Water Regressor...")
        t0 = time.time()
        self.xgb_water.fit(
            X_train, yw_train,
            eval_set=[(X_val, yw_val)],
            verbose=False
        )
        timers["xgb_water"] = time.time() - t0
        yw_pred = self.xgb_water.predict(X_val)
        r2 = r2_score(yw_val, yw_pred)
        rmse = np.sqrt(mean_squared_error(yw_val, yw_pred))
        mae = mean_absolute_error(yw_val, yw_pred)
        self.metrics["xgb_water_r2"] = r2
        self.metrics["xgb_water_rmse"] = rmse
        self.metrics["xgb_water_mae"] = mae
        if verbose:
            print(f"    R²: {r2:.4f} | RMSE: {rmse:.1f} | MAE: {mae:.1f} | Time: {timers['xgb_water']:.1f}s")

        return timers

    def train_random_forest(self, X_train, yc_train, yf_train, yw_train,
                            X_val, yc_val, yf_val, yw_val, verbose=True):
        """Train all Random Forest models."""
        print("\n--- Training Random Forest Models ---")
        timers = {}

        # Crop classification
        if verbose:
            print("\n  [RF] Training Crop Classifier...")
        t0 = time.time()
        self.rf_crop.fit(X_train, yc_train)
        timers["rf_crop"] = time.time() - t0
        yc_pred = self.rf_crop.predict(X_val)
        acc = accuracy_score(yc_val, yc_pred)
        f1 = f1_score(yc_val, yc_pred, average="weighted")
        self.metrics["rf_crop_acc"] = acc
        self.metrics["rf_crop_f1"] = f1
        if verbose:
            print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {timers['rf_crop']:.1f}s")

        # Fertilizer classification
        if verbose:
            print("\n  [RF] Training Fertilizer Classifier...")
        t0 = time.time()
        self.rf_fert.fit(X_train, yf_train)
        timers["rf_fert"] = time.time() - t0
        yf_pred = self.rf_fert.predict(X_val)
        acc = accuracy_score(yf_val, yf_pred)
        f1 = f1_score(yf_val, yf_pred, average="weighted")
        self.metrics["rf_fert_acc"] = acc
        self.metrics["rf_fert_f1"] = f1
        if verbose:
            print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {timers['rf_fert']:.1f}s")

        # Water regression
        if verbose:
            print("\n  [RF] Training Water Regressor...")
        t0 = time.time()
        self.rf_water.fit(X_train, yw_train)
        timers["rf_water"] = time.time() - t0
        yw_pred = self.rf_water.predict(X_val)
        r2 = r2_score(yw_val, yw_pred)
        rmse = np.sqrt(mean_squared_error(yw_val, yw_pred))
        mae = mean_absolute_error(yw_val, yw_pred)
        self.metrics["rf_water_r2"] = r2
        self.metrics["rf_water_rmse"] = rmse
        self.metrics["rf_water_mae"] = mae
        if verbose:
            print(f"    R²: {r2:.4f} | RMSE: {rmse:.1f} | MAE: {mae:.1f} | Time: {timers['rf_water']:.1f}s")

        return timers

    def get_base_predictions(self, X, proba=True):
        """
        Get predictions from all base models.
        Used for generating meta-features for the stacking ensemble.

        Returns dict with probability/prediction arrays.
        """
        preds = {}

        if proba:
            preds["xgb_crop_proba"] = self.xgb_crop.predict_proba(X)
            preds["xgb_fert_proba"] = self.xgb_fert.predict_proba(X)
            preds["rf_crop_proba"] = self.rf_crop.predict_proba(X)
            preds["rf_fert_proba"] = self.rf_fert.predict_proba(X)
        else:
            preds["xgb_crop"] = self.xgb_crop.predict(X)
            preds["xgb_fert"] = self.xgb_fert.predict(X)
            preds["rf_crop"] = self.rf_crop.predict(X)
            preds["rf_fert"] = self.rf_fert.predict(X)

        preds["xgb_water"] = self.xgb_water.predict(X)
        preds["rf_water"] = self.rf_water.predict(X)

        return preds

    def get_feature_importance(self, feature_names):
        """Get feature importance from XGBoost and Random Forest."""
        importance = {}

        # XGBoost feature importance (gain-based)
        if self.xgb_crop is not None:
            imp = self.xgb_crop.feature_importances_
            importance["xgb_crop"] = dict(zip(feature_names, imp))

        if self.rf_crop is not None:
            imp = self.rf_crop.feature_importances_
            importance["rf_crop"] = dict(zip(feature_names, imp))

        return importance

    def save_models(self):
        """Save all traditional models."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        models = {
            "xgb_crop": self.xgb_crop,
            "xgb_fert": self.xgb_fert,
            "xgb_water": self.xgb_water,
            "rf_crop": self.rf_crop,
            "rf_fert": self.rf_fert,
            "rf_water": self.rf_water,
        }

        for name, model in models.items():
            if model is not None:
                path = os.path.join(MODEL_DIR, f"{name}.pkl")
                joblib.dump(model, path)

        print(f"  Traditional models saved to: {MODEL_DIR}")

    def load_models(self):
        """Load all traditional models."""
        model_names = ["xgb_crop", "xgb_fert", "xgb_water",
                       "rf_crop", "rf_fert", "rf_water"]

        for name in model_names:
            path = os.path.join(MODEL_DIR, f"{name}.pkl")
            if os.path.exists(path):
                setattr(self, name, joblib.load(path))

        print(f"  Traditional models loaded from: {MODEL_DIR}")
