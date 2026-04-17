"""
=============================================================================
CONFIGURATION FILE - AI Crop Helper
=============================================================================
Central configuration for all hyperparameters, paths, and model settings.
Modify this file to tune the pipeline without touching model code.

Dataset schema (26 columns):
  Raw features  : N, P, K, temperature, humidity, ph, rainfall,
                  growing_days, Kc, et0_daily, latitude
  Pre-computed  : N_P_ratio, N_K_ratio, P_K_ratio, total_NPK, vpd,
                  heat_stress_index, cold_stress_index, ph_deviation, aridity_index
  Categorical   : soil_type, season, region_climate
  Targets       : crop_label, fertilizer_recommendation, water_requirement_mm

1,920 crop species/varieties · 120,000 training rows · 13 fertilizer types
=============================================================================
"""

import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

DATASET_PATH = os.path.join(DATA_DIR, "crop_dataset_full.csv")

# ============================================================================
# DATA SPLIT
# ============================================================================
TEST_SIZE  = 0.15
VAL_SIZE   = 0.15   # Fraction of train+val set kept for validation
RANDOM_STATE = 42

# ============================================================================
# FEATURE COLUMNS
# ============================================================================
# Raw sensor input features present in the CSV from the Grow Smart device
# (some are computed during data generation but available at inference time)
RAW_FEATURES = [
    "N", "P", "K",
    "temperature", "humidity", "ph", "rainfall",
    "growing_days",   # Days in growing season
    "Kc",             # FAO crop coefficient
    "et0_daily",      # Penman-Monteith ET₀ (mm/day)
    "latitude",       # Approximate latitude band of deployment
]

# Features computed DURING PREPROCESSING (not in CSV — derived at runtime)
ENGINEERED_FEATURES = [
    # Already pre-computed in CSV (pass-through if present)
    "N_P_ratio",          # N / P
    "N_K_ratio",          # N / K
    "P_K_ratio",          # P / K
    "total_NPK",          # N + P + K
    "vpd",                # Vapor pressure deficit (kPa)
    "heat_stress_index",  # max(0, (T - 30) / 10)
    "cold_stress_index",  # max(0, (15 - T) / 10)
    "ph_deviation",       # |pH - 6.5|
    "aridity_index",      # rain / (T + 10)
    # Computed freshly in preprocessing.py
    "nutrient_balance_score",  # Geometric mean of normalized NPK
    "water_stress_index",      # rainfall / (et0_daily * 30)
    "growing_degree_days",     # max(0, T - 10) * growing_days
]

# Categorical features — one-hot encoded
CATEGORICAL_FEATURES = ["soil_type", "season", "region_climate"]

# Target columns
TARGET_CROP       = "crop_label"
TARGET_FERTILIZER = "fertilizer_recommendation"
TARGET_WATER      = "water_requirement_mm"

# ============================================================================
# PYTORCH NEURAL NETWORK HYPERPARAMETERS
# ============================================================================
NN_CONFIG = {
    # Encoder backbone dimensions: input → 512 → 1024 → 512 → 256 → 128
    "hidden_dims": [512, 1024, 512, 256, 128],
    "dropout_rate": 0.25,
    "learning_rate": 8e-4,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "epochs": 120,
    "patience": 20,          # Early stopping patience
    "scheduler_factor": 0.5,
    "scheduler_patience": 10,
    "use_batch_norm": True,
    "activation": "gelu",    # Options: relu, leaky_relu, gelu, silu
    # Multi-task loss weights (also learned via uncertainty weighting)
    "crop_loss_weight": 1.0,
    "fertilizer_loss_weight": 0.85,
    "water_loss_weight": 0.6,
    # Multi-head self-attention
    "use_self_attention": True,
    "attention_heads": 8,    # Must divide hidden_dims[-1]=128 evenly
    # Mixture of Experts
    "n_experts": 4,
    "top_k_experts": 2,
    "expert_dropout": 0.1,
    # Focal loss parameters (for class-imbalanced 1920-class crop problem)
    "focal_gamma": 2.0,
    "focal_alpha": 0.25,
    # Label smoothing (used in cross-entropy heads as fallback)
    "label_smoothing": 0.05,
    # Gradient clipping
    "grad_clip_norm": 1.5,
    # Learning rate warmup
    "warmup_epochs": 5,
    # Mixup augmentation (0 to disable)
    "mixup_alpha": 0.2,
}

# ============================================================================
# XGBOOST HYPERPARAMETERS
# ============================================================================
XGB_CONFIG = {
    "crop": {
        "n_estimators": 500,
        "max_depth": 9,
        "max_leaves": 31,
        "learning_rate": 0.07,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.15,
        "reg_lambda": 1.2,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "early_stopping_rounds": 40,
    },
    "fertilizer": {
        "n_estimators": 500,
        "max_depth": 8,
        "max_leaves": 31,
        "learning_rate": 0.07,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.15,
        "reg_lambda": 1.2,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "early_stopping_rounds": 40,
    },
    "water": {
        "n_estimators": 500,
        "max_depth": 9,
        "max_leaves": 31,
        "learning_rate": 0.07,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.15,
        "reg_lambda": 1.2,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "early_stopping_rounds": 40,
    },
}

# ============================================================================
# RANDOM FOREST HYPERPARAMETERS
# ============================================================================
RF_CONFIG = {
    "crop": {
        "n_estimators": 300,
        "max_depth": 18,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "fertilizer": {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "water": {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 8,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}

# ============================================================================
# ENSEMBLE (STACKING META-LEARNER) HYPERPARAMETERS
# ============================================================================
ENSEMBLE_CONFIG = {
    "meta_learner": "logistic_regression",   # For classification tasks
    "meta_regressor": "ridge",               # For regression (water)
    "cv_folds": 5,
    "use_probabilities": True,               # Use probability outputs as meta-features
}

# ============================================================================
# TRAINING SETTINGS
# ============================================================================
TRAINING_CONFIG = {
    "verbose": True,
    "save_checkpoints": True,
    "log_interval": 10,
    "enable_timers": True,
    "cross_validation_folds": 5,
}
