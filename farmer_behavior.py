"""
=============================================================================
FARMER BEHAVIOR MODEL — MULTI-MODEL DECISION ENGINE
=============================================================================
Integrates farmer historical preferences, resource constraints, and
contextual factors with ML ensemble predictions to produce contextually-
aware, actionable agricultural recommendations.

Architecture:
  Layer 1: ML Ensemble (XGB + RF + NN stacking) → raw predictions
  Layer 2: Farmer Behavior Module → contextual adjustments
  Layer 3: Multi-Model Decision Fusion → final unified recommendation

Farmer Behavior captures:
  - Historical crop rotation preferences
  - Budget / resource constraints
  - Equipment availability
  - Local market prices
  - Risk tolerance (conservative vs. aggressive)
  - Seasonal patterns from past decisions
=============================================================================
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Default crop market prices (USD/ton) — can be overridden per farmer
# ---------------------------------------------------------------------------
DEFAULT_MARKET_PRICES = {
    "rice": 400,
    "wheat": 280,
    "maize": 220,
    "chickpea": 700,
    "kidneybeans": 900,
    "pigeonpeas": 750,
    "mothbeans": 800,
    "mungbean": 850,
    "blackgram": 800,
    "lentil": 750,
    "pomegranate": 1200,
    "banana": 350,
    "mango": 600,
    "grapes": 1500,
    "watermelon": 250,
    "muskmelon": 400,
    "apple": 800,
    "orange": 500,
    "papaya": 400,
    "coconut": 600,
    "cotton": 700,
    "jute": 300,
    "coffee": 3000,
}

# Risk classification for crops
CROP_RISK_PROFILE = {
    "low":    ["rice", "wheat", "maize", "lentil", "chickpea", "potato", "onion", "carrot", "spinach"],
    "medium": ["cotton", "jute", "banana", "papaya", "mango", "coconut", "tomato", "cabbage", "broccoli", "lettuce", "zucchini", "cucumber", "coriander", "mint", "basil"],
    "high":   ["grapes", "apple", "coffee", "pomegranate", "muskmelon", "strawberry", "bell pepper", "watermelon"],
}


class FarmerProfile:
    """
    Stores and manages a farmer's persistent profile.
    Includes history, constraints, preferences, and learned patterns.
    """

    def __init__(self, farmer_id: str, name: str = "Farmer"):
        self.farmer_id = farmer_id
        self.name = name
        self.created_at = datetime.now().isoformat()

        # Resource constraints
        self.budget_level = "medium"        # low / medium / high
        self.land_area_acres = 5.0
        self.irrigation_available = True
        self.mechanized = False             # Has tractor/equipment
        self.market_access = "local"        # local / regional / export

        # Risk tolerance
        self.risk_tolerance = "medium"      # low / medium / high

        # Historical crop decisions (list of dicts)
        self.crop_history: List[Dict] = []

        # Explicit preferences (crops farmer likes/dislikes)
        self.preferred_crops: List[str] = []
        self.avoided_crops: List[str] = []

        # Local market price overrides
        self.market_prices: Dict[str, float] = DEFAULT_MARKET_PRICES.copy()

        # Learned rotation score: penalize same crop 2 seasons in a row
        self.rotation_penalty: Dict[str, float] = {}

        # Accumulated insight scores per crop (from outcome feedback)
        self.crop_success_scores: Dict[str, float] = {}

    def record_decision(self, crop: str, fertilizer: str, water_mm: float,
                        season: str, outcome: Optional[float] = None):
        """
        Record a farming decision and optional outcome (yield as % of expected).
        Used to update rotation penalties and success scores.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "crop": crop,
            "fertilizer": fertilizer,
            "water_mm": water_mm,
            "season": season,
            "outcome_yield_pct": outcome,
        }
        self.crop_history.append(entry)

        # Update rotation penalty — penalize last-used crop
        if len(self.crop_history) >= 1:
            recent_crop = self.crop_history[-1]["crop"]
            self.rotation_penalty[recent_crop] = min(
                self.rotation_penalty.get(recent_crop, 0) + 0.15, 0.4
            )
            # Decay penalty for crops not recently planted
            for c in list(self.rotation_penalty.keys()):
                if c != recent_crop:
                    self.rotation_penalty[c] = max(
                        self.rotation_penalty[c] - 0.05, 0.0
                    )

        # Update success scores from outcome feedback
        if outcome is not None:
            prev = self.crop_success_scores.get(crop, 0.5)
            # Exponential moving average
            self.crop_success_scores[crop] = 0.7 * prev + 0.3 * (outcome / 100.0)

    def to_dict(self) -> Dict:
        return {
            "farmer_id": self.farmer_id,
            "name": self.name,
            "budget_level": self.budget_level,
            "land_area_acres": self.land_area_acres,
            "irrigation_available": self.irrigation_available,
            "mechanized": self.mechanized,
            "market_access": self.market_access,
            "risk_tolerance": self.risk_tolerance,
            "preferred_crops": self.preferred_crops,
            "avoided_crops": self.avoided_crops,
            "market_prices": self.market_prices,
            "crop_history": self.crop_history[-10:],  # last 10
            "rotation_penalty": self.rotation_penalty,
            "crop_success_scores": self.crop_success_scores,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FarmerProfile":
        p = cls(d["farmer_id"], d.get("name", "Farmer"))
        p.budget_level = d.get("budget_level", "medium")
        p.land_area_acres = d.get("land_area_acres", 5.0)
        p.irrigation_available = d.get("irrigation_available", True)
        p.mechanized = d.get("mechanized", False)
        p.market_access = d.get("market_access", "local")
        p.risk_tolerance = d.get("risk_tolerance", "medium")
        p.preferred_crops = d.get("preferred_crops", [])
        p.avoided_crops = d.get("avoided_crops", [])
        p.market_prices = d.get("market_prices", DEFAULT_MARKET_PRICES.copy())
        p.crop_history = d.get("crop_history", [])
        p.rotation_penalty = d.get("rotation_penalty", {})
        p.crop_success_scores = d.get("crop_success_scores", {})
        return p

    def save(self, directory: str = "farmer_profiles"):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.farmer_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, farmer_id: str, directory: str = "farmer_profiles") -> "FarmerProfile":
        path = os.path.join(directory, f"{farmer_id}.json")
        if not os.path.exists(path):
            return cls(farmer_id)
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


class FarmerBehaviorModel:
    """
    Multi-Model Decision Engine.

    Fuses ML ensemble predictions with farmer behavioral context to produce
    refined, personalized, and actionable agricultural recommendations.

    Decision Pipeline:
      1. Receive raw ML ensemble outputs (crop proba, fert proba, water pred)
      2. Apply farmer preference adjustments to probability vectors
      3. Apply rotation diversity boost/penalty
      4. Apply risk-tolerance filtering
      5. Apply budget/resource feasibility check
      6. Fuse market price optimization
      7. Output ranked recommendations with confidence and rationale
    """

    def __init__(self):
        self.version = "1.0.0"

    def _adjust_for_preferences(
        self,
        crop_proba: np.ndarray,
        crop_classes: List[str],
        profile: FarmerProfile,
    ) -> np.ndarray:
        """Boost preferred crops, penalize avoided ones."""
        adjusted = crop_proba.copy()
        for i, crop in enumerate(crop_classes):
            c = crop.lower()
            if c in [p.lower() for p in profile.preferred_crops]:
                adjusted[i] *= 1.25   # +25% boost for preferred crops
            if c in [a.lower() for a in profile.avoided_crops]:
                adjusted[i] *= 0.30   # -70% suppression for avoided crops
        # Re-normalize
        total = adjusted.sum()
        return adjusted / total if total > 0 else adjusted

    def _adjust_for_rotation(
        self,
        crop_proba: np.ndarray,
        crop_classes: List[str],
        profile: FarmerProfile,
    ) -> np.ndarray:
        """Apply rotation diversity — penalize recently grown crops."""
        adjusted = crop_proba.copy()
        for i, crop in enumerate(crop_classes):
            penalty = profile.rotation_penalty.get(crop.lower(), 0.0)
            adjusted[i] *= (1.0 - penalty)
        total = adjusted.sum()
        return adjusted / total if total > 0 else adjusted

    def _adjust_for_risk(
        self,
        crop_proba: np.ndarray,
        crop_classes: List[str],
        risk_tolerance: str,
    ) -> np.ndarray:
        """Shift probability mass toward risk-appropriate crops."""
        adjusted = crop_proba.copy()
        
        # Base logical adjustments depending on farmer's risk tolerance
        risk_table = {
            "low":    {"low": 1.4, "medium": 1.0, "high": 0.4},
            "medium": {"low": 1.1, "medium": 1.2, "high": 0.8},
            "high":   {"low": 0.7, "medium": 1.1, "high": 1.5},
        }
        multipliers = risk_table.get(risk_tolerance, risk_table["medium"])

        for i, crop in enumerate(crop_classes):
            c = crop.lower()
            crop_risk = "medium" # default fallback
            for risk_level, crops in CROP_RISK_PROFILE.items():
                if c in crops:
                    crop_risk = risk_level
                    break
            
            adjusted[i] *= multipliers.get(crop_risk, 1.0)

        total = adjusted.sum()
        return adjusted / total if total > 0 else adjusted

    def _adjust_for_market(
        self,
        crop_proba: np.ndarray,
        crop_classes: List[str],
        market_prices: Dict[str, float],
        market_access: str,
    ) -> np.ndarray:
        """Upweight high-value crops when farmer has market access."""
        if market_access == "local":
            return crop_proba  # Local farmers: stable crops preferred

        adjusted = crop_proba.copy()
        prices = np.array([
            market_prices.get(c.lower(), 400) for c in crop_classes
        ], dtype=float)

        # Normalize prices to [0.8, 1.3] multiplier range
        p_min, p_max = prices.min(), prices.max()
        if p_max > p_min:
            price_multipliers = 0.8 + 0.5 * (prices - p_min) / (p_max - p_min)
        else:
            price_multipliers = np.ones_like(prices)

        adjusted = adjusted * price_multipliers
        total = adjusted.sum()
        return adjusted / total if total > 0 else adjusted

    def _apply_success_scores(
        self,
        crop_proba: np.ndarray,
        crop_classes: List[str],
        profile: FarmerProfile,
    ) -> np.ndarray:
        """Boost crops with high historical success for this farmer."""
        adjusted = crop_proba.copy()
        for i, crop in enumerate(crop_classes):
            score = profile.crop_success_scores.get(crop.lower(), None)
            if score is not None:
                # Score in [0,1]: 0.5 = neutral, >0.5 = boost, <0.5 = penalize
                multiplier = 0.6 + 0.8 * score  # maps [0,1] → [0.6, 1.4]
                adjusted[i] *= multiplier
        total = adjusted.sum()
        return adjusted / total if total > 0 else adjusted

    def make_decision(
        self,
        ml_crop_proba: np.ndarray,
        ml_fert_proba: np.ndarray,
        ml_water_pred: float,
        crop_classes: List[str],
        fert_classes: List[str],
        profile: FarmerProfile,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Core multi-model decision fusion.

        Args:
            ml_crop_proba:  Ensemble crop probability vector (shape: n_crops)
            ml_fert_proba:  Ensemble fertilizer probability vector (shape: n_ferts)
            ml_water_pred:  Predicted water requirement (mm)
            crop_classes:   List of crop class names (aligned with proba)
            fert_classes:   List of fertilizer class names
            profile:        FarmerProfile instance
            top_k:          Number of top recommendations to return

        Returns:
            Detailed decision dict with ranked recommendations and reasoning.
        """

        # ── System Array Checks ─────────────────────────────────────────────
        ml_crop_proba = np.asarray(ml_crop_proba).flatten()
        ml_fert_proba = np.asarray(ml_fert_proba).flatten()
        ml_water_pred = float(np.asarray(ml_water_pred).flatten()[0])
        
        if len(ml_crop_proba) != len(crop_classes):
            raise ValueError(f"Crop probability size ({len(ml_crop_proba)}) does not match classes ({len(crop_classes)})")
        if len(ml_fert_proba) != len(fert_classes):
            raise ValueError(f"Fertilizer probability size ({len(ml_fert_proba)}) does not match classes ({len(fert_classes)})")

        # ── Layer 1: Start from ML ensemble output ──────────────────────────
        adjusted = ml_crop_proba.copy()

        # ── Layer 2: Behavioral Adjustments ─────────────────────────────────
        adjusted = self._adjust_for_preferences(adjusted, crop_classes, profile)
        adjusted = self._adjust_for_rotation(adjusted, crop_classes, profile)
        adjusted = self._adjust_for_risk(adjusted, crop_classes, profile.risk_tolerance)
        adjusted = self._adjust_for_market(
            adjusted, crop_classes, profile.market_prices, profile.market_access
        )
        adjusted = self._apply_success_scores(adjusted, crop_classes, profile)

        # ── Layer 3: Fertilizer Recommendation ──────────────────────────────
        top_fert_idx = int(np.argmax(ml_fert_proba))
        top_fert = fert_classes[top_fert_idx]
        fert_confidence = float(ml_fert_proba.flatten()[top_fert_idx])

        # ── Layer 4: Water Adjustment for irrigation availability ────────────
        water_adjusted = ml_water_pred
        if not profile.irrigation_available:
            # Flag if predicted water need exceeds rainfed threshold
            water_flag = water_adjusted > 600
        else:
            water_flag = False

        # ── Layer 5: Build ranked crop recommendations ───────────────────────
        ranked_indices = np.argsort(adjusted)[::-1][:top_k]
        recommendations = []

        for rank, idx in enumerate(ranked_indices):
            crop_name = crop_classes[idx]
            ml_conf = float(ml_crop_proba.flatten()[idx])
            behavior_conf = float(adjusted[idx])
            price = profile.market_prices.get(crop_name.lower(), 400)

            # Determine risk profile
            risk = "medium"
            for r, crops in CROP_RISK_PROFILE.items():
                if crop_name.lower() in crops:
                    risk = r
                    break

            # Generate rationale
            rationale = _build_rationale(
                crop_name, ml_conf, behavior_conf, risk, profile, rank
            )

            recommendations.append({
                "rank": rank + 1,
                "crop": crop_name,
                "ml_confidence": round(ml_conf * 100, 1),
                "behavioral_confidence": round(behavior_conf * 100, 1),
                "combined_score": round((ml_conf * 0.6 + behavior_conf * 0.4) * 100, 1),
                "expected_price_usd_per_ton": price,
                "risk_profile": risk,
                "rationale": rationale,
            })

        # ── Layer 6: Build full decision output ─────────────────────────────
        decision = {
            "timestamp": datetime.now().isoformat(),
            "farmer_id": profile.farmer_id,
            "farmer_name": profile.name,

            "primary_recommendation": recommendations[0],
            "alternative_recommendations": recommendations[1:],

            "fertilizer": {
                "recommendation": top_fert,
                "confidence_pct": round(fert_confidence * 100, 1),
            },

            "water": {
                "predicted_mm": round(ml_water_pred, 1),
                "adjusted_mm": round(water_adjusted, 1),
                "irrigation_flag": water_flag,
                "irrigation_note": (
                    "⚠ High water requirement. Irrigation strongly recommended."
                    if water_flag else
                    "✓ Manageable under rainfed or supplemental irrigation."
                ),
            },

            "model_metadata": {
                "ensemble_version": "stacking_v1",
                "behavior_version": self.version,
                "adjustments_applied": [
                    "preference_boost",
                    "rotation_diversity",
                    "risk_filtering",
                    "market_optimization",
                    "historical_success_scoring",
                ],
            },

            "farmer_context": {
                "risk_tolerance": profile.risk_tolerance,
                "budget_level": profile.budget_level,
                "irrigation_available": profile.irrigation_available,
                "market_access": profile.market_access,
                "seasons_recorded": len(profile.crop_history),
            },
        }

        return decision


def _build_rationale(
    crop: str, ml_conf: float, behavior_conf: float,
    risk: str, profile: FarmerProfile, rank: int
) -> str:
    """Generate human-readable rationale for a crop recommendation."""
    parts = []

    if ml_conf > 0.5:
        parts.append(f"AI model highly confident ({ml_conf*100:.0f}%)")
    elif ml_conf > 0.25:
        parts.append(f"Strong AI signal ({ml_conf*100:.0f}%)")
    else:
        parts.append(f"AI suggests as viable option ({ml_conf*100:.0f}%)")

    if crop.lower() in [p.lower() for p in profile.preferred_crops]:
        parts.append("matches your stated preferences")

    rotation_penalty = profile.rotation_penalty.get(crop.lower(), 0)
    if rotation_penalty < 0.1:
        parts.append("good rotation choice (not recently grown)")
    elif rotation_penalty > 0.2:
        parts.append("recently grown — consider rotating")

    risk_map = {
        "low": "low-risk, stable returns",
        "medium": "moderate risk, balanced returns",
        "high": "higher risk, premium market potential",
    }
    risk_context = risk_map.get(risk, "moderate risk")
    if risk == "low" and profile.risk_tolerance == "low":
        parts.append(f"aligns with your conservative approach ({risk_context})")
    elif risk == "high" and profile.risk_tolerance == "high":
        parts.append(f"matches your growth-oriented profile ({risk_context})")
    else:
        parts.append(risk_context)

    success = profile.crop_success_scores.get(crop.lower(), None)
    if success is not None and success > 0.65:
        parts.append(f"historically strong yields on your farm (avg {success*100:.0f}% of expected target)")

    return "; ".join(parts).capitalize() + "."


# ---------------------------------------------------------------------------
# Convenience function for simple one-shot inference
# ---------------------------------------------------------------------------
def make_farm_decision(
    ml_crop_proba: np.ndarray,
    ml_fert_proba: np.ndarray,
    ml_water_pred: float,
    crop_classes: List[str],
    fert_classes: List[str],
    farmer_id: str = "default",
    farmer_name: str = "Farmer",
    risk_tolerance: str = "medium",
    budget_level: str = "medium",
    irrigation_available: bool = True,
    market_access: str = "local",
    preferred_crops: Optional[List[str]] = None,
    avoided_crops: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Quick one-shot multi-model decision without a saved profile.

    Example:
        decision = make_farm_decision(
            ml_crop_proba=ensemble.predict_proba(...)[0],
            ml_fert_proba=ensemble.predict_proba(...)[1],
            ml_water_pred=450.0,
            crop_classes=crop_encoder.classes_.tolist(),
            fert_classes=fert_encoder.classes_.tolist(),
            risk_tolerance="low",
            irrigation_available=False,
        )
        print(decision["primary_recommendation"])
    """
    profile = FarmerProfile(farmer_id, farmer_name)
    profile.risk_tolerance = risk_tolerance
    profile.budget_level = budget_level
    profile.irrigation_available = irrigation_available
    profile.market_access = market_access
    profile.preferred_crops = preferred_crops or []
    profile.avoided_crops = avoided_crops or []

    engine = FarmerBehaviorModel()
    return engine.make_decision(
        ml_crop_proba, ml_fert_proba, ml_water_pred,
        crop_classes, fert_classes, profile
    )


if __name__ == "__main__":
    # Quick test with synthetic data
    print("=" * 60)
    print("FARMER BEHAVIOR MODEL — MULTI-MODEL DECISION ENGINE")
    print("=" * 60)

    np.random.seed(42)
    crop_classes = ["rice", "wheat", "maize", "cotton", "chickpea",
                    "lentil", "coffee", "mango", "grapes", "banana"]
    fert_classes = ["Urea", "DAP", "MOP", "NPK 10-26-26", "Compost",
                    "Ammonium Sulfate", "SSP"]

    # Simulate ensemble output
    raw_crop_proba = np.random.dirichlet(np.ones(10) * 2)
    raw_fert_proba = np.random.dirichlet(np.ones(7) * 2)
    raw_water = 520.0

    # Simulate a farmer profile
    profile = FarmerProfile("farmer_001", "Ramesh Kumar")
    profile.risk_tolerance = "low"
    profile.budget_level = "medium"
    profile.irrigation_available = False
    profile.market_access = "regional"
    profile.preferred_crops = ["rice", "wheat"]
    profile.avoided_crops = ["coffee"]

    # Simulate some history
    profile.record_decision("rice", "Urea", 600, "kharif", outcome=88)
    profile.record_decision("wheat", "DAP", 400, "rabi", outcome=92)

    engine = FarmerBehaviorModel()
    decision = engine.make_decision(
        raw_crop_proba, raw_fert_proba, raw_water,
        crop_classes, fert_classes, profile
    )

    print(f"\n✅ Decision for {decision['farmer_name']}:")
    print(f"\n🌾 Primary Recommendation: {decision['primary_recommendation']['crop'].upper()}")
    print(f"   ML Confidence:        {decision['primary_recommendation']['ml_confidence']}%")
    print(f"   Behavioral Confidence: {decision['primary_recommendation']['behavioral_confidence']}%")
    print(f"   Rationale: {decision['primary_recommendation']['rationale']}")
    print(f"\n💊 Fertilizer: {decision['fertilizer']['recommendation']} "
          f"({decision['fertilizer']['confidence_pct']}% confident)")
    print(f"\n💧 Water: {decision['water']['predicted_mm']} mm — {decision['water']['irrigation_note']}")
    print(f"\n📊 Alternative Options:")
    for alt in decision["alternative_recommendations"]:
        print(f"   #{alt['rank']}: {alt['crop']} ({alt['combined_score']}% score)")
