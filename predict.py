"""
=============================================================================
UNIFIED INFERENCE ENGINE — AI Crop Helper
=============================================================================
Single-entry-point inference: sensor readings → multi-model decision.

Usage:
    from predict import CropPredictor

    predictor = CropPredictor()
    predictor.load()

    result = predictor.predict(
        N=90, P=42, K=43,
        temperature=20.9, humidity=82.0, ph=6.5, rainfall=202.9,
        farmer_id="farmer_001",
        risk_tolerance="medium",
    )
    print(result)
=============================================================================
"""

import numpy as np
import joblib
import torch
import os
import json
from typing import Dict, Any, Optional, List

from config import MODEL_DIR, RAW_FEATURES, ENGINEERED_FEATURES
from farmer_behavior import FarmerBehaviorModel, FarmerProfile, make_farm_decision


# ---------------------------------------------------------------------------
# Optional: import NN model if available
# ---------------------------------------------------------------------------
try:
    from models_nn import CropMultiTaskNet
    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False


class CropPredictor:
    """
    Unified inference engine for AI Crop Helper.

    Loads all trained models (XGB, RF, NN, Ensemble) and the preprocessor,
    then runs the full decision pipeline including farmer behavior fusion.
    """

    def __init__(self):
        self.preprocessor = None
        self.traditional_models = None
        self.nn_model = None
        self.ensemble = None
        self.behavior_engine = FarmerBehaviorModel()
        self._loaded = False

    def load(self, model_dir: str = None) -> "CropPredictor":
        """Load all model artifacts from disk."""
        model_dir = model_dir or MODEL_DIR
        print("Loading AI Crop Helper models...")

        # ── Preprocessor ────────────────────────────────────────────────────
        prep_path = os.path.join(model_dir, "preprocessor.pkl")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {prep_path}. "
                "Run train.py first to generate models."
            )
        self.preprocessor = joblib.load(prep_path)
        print(f"  ✓ Preprocessor loaded "
              f"({self.preprocessor['n_features']} features, "
              f"{self.preprocessor['n_crop_classes']} crop classes)")

        # ── Traditional Models (XGB + RF) ────────────────────────────────────
        from models_traditional import TraditionalModels
        self.traditional_models = TraditionalModels()
        self.traditional_models.load_models()

        # ── Neural Network ───────────────────────────────────────────────────
        if NN_AVAILABLE:
            nn_path = os.path.join(model_dir, "nn_model.pt")
            if os.path.exists(nn_path):
                self.nn_model = CropMultiTaskNet(
                    n_features=self.preprocessor["n_features"],
                    n_crop_classes=self.preprocessor["n_crop_classes"],
                    n_fert_classes=self.preprocessor["n_fert_classes"],
                )
                self.nn_model.load_state_dict(
                    torch.load(nn_path, map_location="cpu", weights_only=True)
                )
                self.nn_model.eval()
                print("  ✓ Neural network loaded")
            else:
                print("  ⚠ NN model file not found — skipping NN")
                self.nn_model = None

        # ── Ensemble Meta-Learner ─────────────────────────────────────────────
        from ensemble import StackingEnsemble
        self.ensemble = StackingEnsemble(
            n_crop_classes=self.preprocessor["n_crop_classes"],
            n_fert_classes=self.preprocessor["n_fert_classes"],
        )
        self.ensemble.load()

        self._loaded = True
        print("  ✓ All models loaded successfully\n")
        return self

    def _preprocess(self, N: float, P: float, K: float,
                    temperature: float, humidity: float,
                    ph: float, rainfall: float) -> np.ndarray:
        """Transform raw sensor readings into model-ready feature vector."""
        import pandas as pd

        # Build raw feature row
        row = {
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }

        # Engineer features
        row["N_P_ratio"]          = N / (P + 1e-6)
        row["N_K_ratio"]          = N / (K + 1e-6)
        row["P_K_ratio"]          = P / (K + 1e-6)
        row["total_NPK"]          = N + P + K
        row["vpd"]                = (0.6108 * np.exp(17.27 * temperature / (temperature + 237.3)) *
                                     (1 - humidity / 100))
        row["heat_moisture_index"] = (temperature + 10) / (rainfall / 100 + 1)
        row["aridity_index"]       = rainfall / (temperature + 10 + 1e-6)

        # Build numeric array (no categorical features for inference default)
        feature_order = RAW_FEATURES + ENGINEERED_FEATURES
        x = np.array([[row[f] for f in feature_order]], dtype=np.float32)

        # Scale
        scaler = self.preprocessor["scaler"]
        # Scaler was trained on full feature set; pad with zeros for
        # categorical features if present, then select first n columns
        n_scale_features = scaler.center_.shape[0]
        if x.shape[1] < n_scale_features:
            pad = np.zeros((1, n_scale_features - x.shape[1]))
            x_padded = np.hstack([x, pad])
        else:
            x_padded = x[:, :n_scale_features]

        x_scaled = scaler.transform(x_padded)
        return x_scaled

    def predict_raw(self, N: float, P: float, K: float,
                    temperature: float, humidity: float,
                    ph: float, rainfall: float) -> Dict[str, Any]:
        """
        Run the full ML ensemble on raw sensor readings.
        Returns raw model outputs (pre-behavior fusion).
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call predictor.load() first.")

        X = self._preprocess(N, P, K, temperature, humidity, ph, rainfall)

        crop_encoder = self.preprocessor["crop_encoder"]
        fert_encoder = self.preprocessor["fert_encoder"]

        # ── Get base predictions ─────────────────────────────────────────────
        if self.nn_model is not None:
            # Full ensemble with NN
            from ensemble import StackingEnsemble
            crop_proba, fert_proba = self.ensemble.predict_proba(
                self.traditional_models, self.nn_model, X
            )
            _, _, water_pred = self.ensemble.predict(
                self.traditional_models, self.nn_model, X
            )
            water_val = float(water_pred[0]) if hasattr(water_pred, '__len__') else float(water_pred)
        else:
            # Fallback: use XGB only
            crop_proba = self.traditional_models.xgb_crop.predict_proba(X)
            fert_proba = self.traditional_models.xgb_fert.predict_proba(X)
            water_val = float(self.traditional_models.xgb_water.predict(X)[0])

        return {
            "crop_proba": crop_proba[0],
            "fert_proba": fert_proba[0],
            "water_pred": water_val,
            "crop_classes": crop_encoder.classes_.tolist(),
            "fert_classes": fert_encoder.classes_.tolist(),
        }

    def predict(
        self,
        N: float, P: float, K: float,
        temperature: float, humidity: float,
        ph: float, rainfall: float,
        # Farmer context (optional)
        farmer_id: str = "default",
        farmer_name: str = "Farmer",
        risk_tolerance: str = "medium",
        budget_level: str = "medium",
        irrigation_available: bool = True,
        market_access: str = "local",
        preferred_crops: Optional[List[str]] = None,
        avoided_crops: Optional[List[str]] = None,
        load_profile: bool = False,
    ) -> Dict[str, Any]:
        """
        Full prediction pipeline: sensor readings → multi-model decision.

        Args:
            N, P, K:             Soil macronutrient levels (mg/kg)
            temperature:          Ambient temperature (°C)
            humidity:             Relative humidity (%)
            ph:                  Soil pH (0–14)
            rainfall:            Annual rainfall (mm)
            farmer_id:           Unique farmer identifier
            farmer_name:         Display name
            risk_tolerance:      'low' | 'medium' | 'high'
            budget_level:        'low' | 'medium' | 'high'
            irrigation_available: Whether irrigation is accessible
            market_access:       'local' | 'regional' | 'export'
            preferred_crops:     Crops the farmer prefers
            avoided_crops:       Crops the farmer wants to avoid
            load_profile:        If True, load existing profile from disk

        Returns:
            Complete decision dict with ranked recommendations, fertilizer,
            water requirement, and full rationale.
        """
        # Step 1: ML ensemble
        raw = self.predict_raw(N, P, K, temperature, humidity, ph, rainfall)

        # Step 2: Farmer profile
        if load_profile:
            profile = FarmerProfile.load(farmer_id)
        else:
            profile = FarmerProfile(farmer_id, farmer_name)
            profile.risk_tolerance = risk_tolerance
            profile.budget_level = budget_level
            profile.irrigation_available = irrigation_available
            profile.market_access = market_access
            profile.preferred_crops = preferred_crops or []
            profile.avoided_crops = avoided_crops or []

        # Step 3: Multi-model behavioral fusion
        decision = self.behavior_engine.make_decision(
            ml_crop_proba=raw["crop_proba"],
            ml_fert_proba=raw["fert_proba"],
            ml_water_pred=raw["water_pred"],
            crop_classes=raw["crop_classes"],
            fert_classes=raw["fert_classes"],
            profile=profile,
        )

        # Step 4: Add sensor data to output
        decision["sensor_input"] = {
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }

        return decision

    def get_model_metrics(self) -> Dict[str, Any]:
        """Return training metrics from saved logs."""
        log_path = os.path.join(os.path.dirname(MODEL_DIR), "logs", "training_results.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                return json.load(f)
        return {}


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("  AI CROP HELPER — UNIFIED INFERENCE ENGINE")
    print("=" * 65)

    predictor = CropPredictor()
    predictor.load()

    # Example: sensor reading from a rice-suitable field
    print("\n📡 Sensor Reading: Rice-Suitable Soil")
    print("-" * 45)
    result = predictor.predict(
        N=90, P=42, K=43,
        temperature=20.9, humidity=82.0, ph=6.5, rainfall=202.9,
        farmer_id="demo_farmer",
        farmer_name="Arjun Singh",
        risk_tolerance="low",
        budget_level="medium",
        irrigation_available=True,
        market_access="regional",
    )

    print(f"\n🌾 TOP RECOMMENDATION: {result['primary_recommendation']['crop'].upper()}")
    print(f"   Combined Score:  {result['primary_recommendation']['combined_score']}%")
    print(f"   ML Confidence:   {result['primary_recommendation']['ml_confidence']}%")
    print(f"   Rationale:       {result['primary_recommendation']['rationale']}")
    print(f"\n💊 FERTILIZER: {result['fertilizer']['recommendation']} "
          f"({result['fertilizer']['confidence_pct']}% confidence)")
    print(f"\n💧 WATER REQUIREMENT: {result['water']['predicted_mm']} mm")
    print(f"   {result['water']['irrigation_note']}")
    print(f"\n📋 ALTERNATIVES:")
    for alt in result["alternative_recommendations"]:
        print(f"   #{alt['rank']}: {alt['crop']} — {alt['combined_score']}% score")

    print(f"\n📊 MODEL METRICS:")
    metrics = predictor.get_model_metrics()
    if metrics:
        ens = metrics.get("ensemble", {})
        print(f"   Crop Accuracy:  {ens.get('crop_acc', 0)*100:.1f}%")
        print(f"   Fert Accuracy:  {ens.get('fert_acc', 0)*100:.1f}%")
        print(f"   Water R²:       {ens.get('water_r2', 0):.4f}")
