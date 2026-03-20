# 🌱 AI Crop Helper — Precision Agriculture Intelligence

> **Multi-model AI system** that transforms $89 soil sensor readings into actionable crop, fertilizer, and water recommendations — powered by a stacking ensemble of XGBoost, Random Forest, and PyTorch Neural Networks fused with farmer behavioral intelligence.

[![Crop Accuracy](https://img.shields.io/badge/Crop%20Accuracy-98.8%25-brightgreen)](logs/training_results.json)
[![Fert Accuracy](https://img.shields.io/badge/Fertilizer%20Accuracy-90.5%25-green)](logs/training_results.json)
[![Water R²](https://img.shields.io/badge/Water%20R²-0.953-blue)](logs/training_results.json)
[![Seed Round](https://img.shields.io/badge/Raising-Seed%20Round%202026-orange)](website/index.html)

---

## 🎯 What It Does

The sensor measures **N, P, K, temperature, humidity, pH, and rainfall** from the soil. In under 2 seconds, the AI delivers:

1. **Crop recommendation** — Which crop to plant (22 types), with confidence score
2. **Fertilizer recommendation** — Exact fertilizer type (7 options), dosage guidance
3. **Water requirement** — Predicted mm needed for the season
4. **Farmer-personalized decisions** — Fused with behavioral AI that learns the farmer's history, risk tolerance, budget, and market access

---

## 🧠 Architecture

```
Sensor Input (7 features)
       │
       ▼
Feature Engineering (14 features: NPK ratios, VPD, aridity index, etc.)
       │
   ┌───┴────────────────────────────────┐
   │                                    │
   ▼                                    ▼
XGBoost Models              Random Forest Models
(Crop / Fert / Water)       (Crop / Fert / Water)
   │                                    │
   └───────────────┬────────────────────┘
                   │
                   ▼
         PyTorch Neural Network
        (Multi-task attention, 256→512→256→128)
                   │
                   ▼
       Stacking Ensemble Meta-Learner
       (Logistic Regression + Ridge)
                   │
                   ▼
       Farmer Behavior Model
       (Rotation / Risk / Budget / History)
                   │
                   ▼
        Unified Decision Output
    (Crop + Fertilizer + Water + Rationale)
```

---

## 📊 Model Performance

| Model | Crop Acc | Fert Acc | Water R² |
|-------|----------|----------|----------|
| XGBoost | 98.7% | 90.4% | 0.953 |
| Random Forest | 98.8% | 89.9% | 0.955 |
| Neural Network | 97.4% | 41.7%* | 0.949 |
| **Ensemble (final)** | **98.8%** | **90.5%** | **0.946** |

*NN fertilizer accuracy improves significantly with larger datasets. Ensemble uses meta-learning to correct this.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install numpy pandas torch scikit-learn xgboost joblib
```

### 2. Run inference
```python
from predict import CropPredictor

predictor = CropPredictor()
predictor.load()

result = predictor.predict(
    N=90, P=42, K=43,
    temperature=20.9, humidity=82.0,
    ph=6.5, rainfall=202.9,
    farmer_id="farmer_001",
    risk_tolerance="low",
    irrigation_available=True,
)

print(result["primary_recommendation"])
# → {'crop': 'rice', 'combined_score': 94.2, 'rationale': '...'}
```

### 3. Use farmer behavior engine directly
```python
from farmer_behavior import make_farm_decision
import numpy as np

decision = make_farm_decision(
    ml_crop_proba=np.array([0.7, 0.1, 0.05, ...]),
    ml_fert_proba=np.array([0.4, 0.3, 0.1, ...]),
    ml_water_pred=520.0,
    crop_classes=["rice", "wheat", "maize", ...],
    fert_classes=["Urea", "DAP", "MOP", ...],
    risk_tolerance="medium",
    irrigation_available=False,
    preferred_crops=["rice", "wheat"],
)
```

---

## 📁 Project Structure

```
AI Crop Helper/
├── website/
│   └── index.html          # ← VC-ready investor website
│
├── models/                 # Trained model artifacts
│   ├── preprocessor.pkl    # Scaler + label encoders
│   ├── xgb_crop.pkl        # XGBoost crop classifier
│   ├── xgb_fert.pkl        # XGBoost fertilizer classifier
│   ├── xgb_water.pkl       # XGBoost water regressor
│   ├── rf_crop.pkl         # Random Forest crop classifier
│   ├── rf_fert.pkl         # Random Forest fertilizer classifier
│   ├── rf_water.pkl        # Random Forest water regressor
│   ├── nn_model.pt         # PyTorch neural network weights
│   └── ensemble.pkl        # Stacking meta-learner
│
├── logs/
│   ├── training_results.json  # All model metrics
│   └── training_timing.json   # Training time breakdown
│
├── farmer_behavior.py      # ← Farmer behavior + multi-model decision engine
├── predict.py              # ← Unified inference engine (start here)
├── ensemble.py             # Stacking ensemble implementation
├── models_traditional.py   # XGBoost + Random Forest wrappers
├── models_nn.py            # PyTorch multi-task network
├── preprocessing.py        # Feature engineering + data pipeline
├── train.py                # Full training script
├── config.py               # All hyperparameters
└── requirements.txt
```

---

## 🌐 Deploy the Website

### Option A: GitHub Pages (Recommended — Free, instant)

```bash
# 1. Create a new GitHub repo
git init
git add .
git commit -m "Initial commit — AI Crop Helper"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-crop-helper.git
git push -u origin main

# 2. In GitHub: Settings → Pages → Source: Deploy from branch → Branch: main → /website
# 3. Your site goes live at: https://YOUR_USERNAME.github.io/ai-crop-helper/
```

**Or even simpler — put index.html in root `/docs` folder:**
```bash
mkdir docs
cp website/index.html docs/index.html
git add docs/
git commit -m "Add website to docs folder"
git push
# Then in GitHub Pages settings, set source to /docs folder
```

### Option B: Vercel (Zero-config, custom domain support)

```bash
# Install Vercel CLI
npm install -g vercel

# From project root
cd "website"
vercel deploy

# Follow prompts → get instant URL like https://ai-crop-helper.vercel.app
# Connect GitHub repo for auto-deploy on every push
```

### Option C: Netlify (Drag and drop — 30 seconds)

1. Go to [netlify.com](https://netlify.com) → Sign up free
2. Drag the `website/` folder onto the Netlify dashboard
3. Get a live URL instantly (e.g., `https://ai-crop-helper.netlify.app`)
4. Connect GitHub for automatic deploys

### Option D: Custom Domain

After deploying to Vercel or Netlify:
1. Buy domain (e.g., `aicrophelper.com`) on Namecheap (~$12/year)
2. In Vercel/Netlify settings → Custom Domains → Add your domain
3. Update DNS records as instructed — live in 24 hours

---

## 🔧 Retrain Models

```bash
# Ensure data is at data/crop_dataset_full.csv
python train.py

# This will:
# - Preprocess and engineer features
# - Train XGBoost, Random Forest, Neural Network
# - Build stacking ensemble
# - Save all models to models/
# - Save metrics to logs/
```

---

## 📡 API Server (Optional — for connecting website to live model)

Install FastAPI:
```bash
pip install fastapi uvicorn
```

Create `api.py` and run:
```bash
uvicorn api:app --reload --port 8000
# POST /predict → { N, P, K, temperature, humidity, ph, rainfall }
```

Deploy API to [Railway](https://railway.app) or [Render](https://render.com) — both have free tiers.

---

## 🤝 Investor Contact

- 📧 bindalfamjam@gmail.com
- 🌐 [AI Crop Helper Website](website/index.html)
- 📊 [Technical Research Paper](AI_Crop_Helper_Technical_Paper.pdf)

---

## 📄 License

Proprietary — All rights reserved. Contact for licensing inquiries.

---

*"Feed the world with intelligence."* 🌍
