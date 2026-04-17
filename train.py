"""
=============================================================================
MAIN TRAINING PIPELINE - AI Crop Helper
=============================================================================
ONE-CLICK TRAINING: Just run `python train.py` to start everything.

Pipeline Order:
  1. Data loading & preprocessing
  2. Train XGBoost models (Crop, Fertilizer, Water)
  3. Train Random Forest models (Crop, Fertilizer, Water)
  4. Train PyTorch Multi-Task Neural Network
  5. Train Stacking Ensemble Meta-Learner
  6. Evaluate all models on test set
  7. Generate comparison report
  8. Save all models and artifacts

All steps are timed. Progress is printed in real-time.
=============================================================================
"""

import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix
)
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    NN_CONFIG, MODEL_DIR, LOG_DIR, TRAINING_CONFIG,
    RANDOM_STATE
)
from preprocessing import DataPreprocessor
from models_nn import CropMultiTaskNet, MultiTaskLoss, print_model_summary, mixup_data
from models_traditional import TraditionalModels
from ensemble import StackingEnsemble


# ============================================================================
# TIMER UTILITY
# ============================================================================

class TrainingTimer:
    """Tracks and displays training time for each phase."""

    def __init__(self):
        self.timers = {}
        self.phase_start = None
        self.global_start = None

    def start_global(self):
        self.global_start = time.time()
        print(f"\n{'='*60}")
        print(f"  TRAINING STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

    def start_phase(self, name):
        self.phase_start = time.time()
        print(f"\n{'─'*60}")
        print(f"  PHASE: {name}")
        print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*60}")

    def end_phase(self, name):
        if self.phase_start is None:
            elapsed = 0.0
        else:
            elapsed = time.time() - self.phase_start
        self.timers[name] = elapsed
        print(f"\n  ✓ {name} completed in {self._format_time(elapsed)}")

    def end_global(self):
        if self.global_start is None:
            total = 0.0
        else:
            total = time.time() - self.global_start
        self.timers["TOTAL"] = total

        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"\n  TIMING SUMMARY:")
        print(f"  {'Phase':<35} {'Time':>10}")
        print(f"  {'─'*45}")
        for phase, t in self.timers.items():
            if phase != "TOTAL":
                pct = (t / total) * 100 if total else 0.0
                print(f"  {phase:<35} {self._format_time(t):>10} ({pct:.1f}%)")
        print(f"  {'─'*45}")
        print(f"  {'TOTAL':<35} {self._format_time(total):>10}")
        print(f"{'='*60}\n")

    def _format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            m, s = divmod(seconds, 60)
            return f"{int(m)}m {s:.1f}s"
        else:
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{int(h)}h {int(m)}m {s:.0f}s"

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.timers, f, indent=2)


# ============================================================================
# NEURAL NETWORK TRAINER
# ============================================================================

class NNTrainer:
    """Handles the PyTorch training loop with early stopping and logging."""

    def __init__(self, model, config=None):
        if config is None:
            config = NN_CONFIG

        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss function
        self.criterion = MultiTaskLoss(
            model.n_crop_classes, model.n_fert_classes, config
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

        # Learning rate scheduler with linear warmup + cosine annealing
        warmup_epochs = config.get("warmup_epochs", 5)
        self.warmup_epochs = warmup_epochs
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        # Training history
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_crop_acc": [], "val_crop_acc": [],
            "train_fert_acc": [], "val_fert_acc": [],
            "val_water_r2": [],
        }

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def train_epoch(self, train_loader, use_mixup=True):
        """Train for one epoch with optional Mixup augmentation."""
        self.model.train()
        total_loss = 0
        crop_correct = 0
        fert_correct = 0
        n_samples = 0
        mixup_alpha = self.config.get("mixup_alpha", 0.2)

        for X, yc, yf, yw in train_loader:
            X = X.to(self.device)
            yc = yc.to(self.device)
            yf = yf.to(self.device)
            yw = yw.to(self.device)

            self.optimizer.zero_grad()

            if use_mixup and mixup_alpha > 0:
                X_mix, yc_a, yc_b, yf_a, yf_b, yw_mix, lam = mixup_data(
                    X, yc, yf, yw, alpha=mixup_alpha
                )
                crop_logits, fert_logits, water_pred, _, _, moe_aux = self.model(X_mix)
                # Mixup loss: weighted combination of losses for both label sets
                loss_a, _ = self.criterion(
                    crop_logits, fert_logits, water_pred, yc_a, yf_a, yw_mix,
                    moe_aux_loss=moe_aux
                )
                loss_b, _ = self.criterion(
                    crop_logits, fert_logits, water_pred, yc_b, yf_b, yw_mix,
                    moe_aux_loss=moe_aux
                )
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                crop_logits, fert_logits, water_pred, _, _, moe_aux = self.model(X)
                loss, _ = self.criterion(
                    crop_logits, fert_logits, water_pred, yc, yf, yw,
                    moe_aux_loss=moe_aux
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get("grad_clip_norm", 1.5)
            )
            self.optimizer.step()

            total_loss += loss.item() * X.size(0)
            crop_correct += (crop_logits.argmax(1) == yc).sum().item()
            fert_correct += (fert_logits.argmax(1) == yf).sum().item()
            n_samples += X.size(0)

        self.scheduler.step()

        return {
            "loss": total_loss / n_samples,
            "crop_acc": crop_correct / n_samples,
            "fert_acc": fert_correct / n_samples,
        }

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        crop_correct = 0
        fert_correct = 0
        n_samples = 0
        all_water_true = []
        all_water_pred = []

        for X, yc, yf, yw in val_loader:
            X = X.to(self.device)
            yc = yc.to(self.device)
            yf = yf.to(self.device)
            yw = yw.to(self.device)

            crop_logits, fert_logits, water_pred, _, _, moe_aux = self.model(X)
            loss, loss_dict = self.criterion(
                crop_logits, fert_logits, water_pred, yc, yf, yw,
                moe_aux_loss=moe_aux
            )

            total_loss += loss.item() * X.size(0)
            crop_correct += (crop_logits.argmax(1) == yc).sum().item()
            fert_correct += (fert_logits.argmax(1) == yf).sum().item()
            n_samples += X.size(0)

            all_water_true.extend(yw.cpu().numpy())
            all_water_pred.extend(water_pred.squeeze().cpu().numpy())

        water_r2 = r2_score(all_water_true, all_water_pred)

        return {
            "loss": total_loss / n_samples,
            "crop_acc": crop_correct / n_samples,
            "fert_acc": fert_correct / n_samples,
            "water_r2": water_r2,
        }

    def train(self, train_loader, val_loader, epochs=None, verbose=True):
        """Full training loop with early stopping."""
        if epochs is None:
            epochs = self.config["epochs"]
        patience = self.config["patience"]
        log_interval = TRAINING_CONFIG.get("log_interval", 10)

        print(f"\n  [NN] Training for up to {epochs} epochs (patience={patience})")
        print(f"  Device: {self.device}")

        total_params, trainable_params = self.model.get_num_parameters()
        print(f"  Parameters: {trainable_params:,} trainable / {total_params:,} total")

        t0 = time.time()

        epoch = 0
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_crop_acc"].append(train_metrics["crop_acc"])
            self.history["val_crop_acc"].append(val_metrics["crop_acc"])
            self.history["train_fert_acc"].append(train_metrics["fert_acc"])
            self.history["val_fert_acc"].append(val_metrics["fert_acc"])
            self.history["val_water_r2"].append(val_metrics["water_r2"])

            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                marker = " ★"
            else:
                self.patience_counter += 1
                marker = ""

            # Logging
            epoch_time = time.time() - epoch_start
            if verbose and (epoch % log_interval == 0 or epoch == 1 or marker):
                lr = self.optimizer.param_groups[0]['lr']
                # Log uncertainty sigmas from the loss function
                sigma_info = ""
                if hasattr(self.criterion, 'log_var_crop'):
                    import torch as _t
                    sc = _t.exp(0.5 * self.criterion.log_var_crop).item()
                    sf = _t.exp(0.5 * self.criterion.log_var_fert).item()
                    sw = _t.exp(0.5 * self.criterion.log_var_water).item()
                    sigma_info = f" │ σ: {sc:.2f}/{sf:.2f}/{sw:.2f}"
                print(f"  Epoch {epoch:>4}/{epochs} │ "
                      f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} │ "
                      f"Crop: {val_metrics['crop_acc']:.3f} │ "
                      f"Fert: {val_metrics['fert_acc']:.3f} │ "
                      f"Water R²: {val_metrics['water_r2']:.3f} │ "
                      f"LR: {lr:.6f}{sigma_info} │ "
                      f"{epoch_time:.1f}s{marker}")

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"  Restored best model (val_loss={self.best_val_loss:.4f})")

        total_time = time.time() - t0
        print(f"  NN training completed in {total_time:.1f}s ({epoch} epochs)")

        return total_time

    def save_model(self):
        """Save the PyTorch model and training history."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Save model state dict directly (compatible with predict.py load)
        model_path = os.path.join(MODEL_DIR, "nn_model.pt")
        torch.save(self.model.state_dict(), model_path)

        # Save training history
        hist_path = os.path.join(MODEL_DIR, "nn_history.json")
        with open(hist_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"  NN model saved to: {model_path}")

    def get_test_predictions(self, X_test):
        """Get predictions on test data."""
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            X = torch.FloatTensor(X_test).to(self.device)
            crop_logits, fert_logits, water_pred, gates, attn, _ = self.model(X)

            crop_pred = crop_logits.argmax(1).cpu().numpy()
            fert_pred = fert_logits.argmax(1).cpu().numpy()
            water = water_pred.squeeze().cpu().numpy()
            crop_proba = torch.softmax(crop_logits, dim=1).cpu().numpy()

        return crop_pred, fert_pred, water, crop_proba


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_all_models(data, traditional_models, nn_trainer, ensemble, timer):
    """Comprehensive evaluation of all models on the test set."""
    timer.start_phase("Final Evaluation on Test Set")

    X_test = data["X_test"]
    yc_test = data["yc_test"]
    yf_test = data["yf_test"]
    yw_test = data["yw_test"]
    crop_encoder = data["crop_encoder"]
    fert_encoder = data["fert_encoder"]

    results = {}

    print("\n  === TEST SET RESULTS ===\n")

    # --- XGBoost ---
    print("  [XGBoost]")
    xgb_crop = traditional_models.xgb_crop.predict(X_test)
    xgb_fert = traditional_models.xgb_fert.predict(X_test)
    xgb_water = traditional_models.xgb_water.predict(X_test)

    results["xgb"] = {
        "crop_acc": accuracy_score(yc_test, xgb_crop),
        "crop_f1": f1_score(yc_test, xgb_crop, average="weighted"),
        "fert_acc": accuracy_score(yf_test, xgb_fert),
        "fert_f1": f1_score(yf_test, xgb_fert, average="weighted"),
        "water_r2": r2_score(yw_test, xgb_water),
        "water_rmse": np.sqrt(mean_squared_error(yw_test, xgb_water)),
        "water_mae": mean_absolute_error(yw_test, xgb_water),
    }
    print(f"    Crop Acc:  {results['xgb']['crop_acc']:.4f}  |  F1: {results['xgb']['crop_f1']:.4f}")
    print(f"    Fert Acc:  {results['xgb']['fert_acc']:.4f}  |  F1: {results['xgb']['fert_f1']:.4f}")
    print(f"    Water R²:  {results['xgb']['water_r2']:.4f}  |  RMSE: {results['xgb']['water_rmse']:.1f}")

    # --- Random Forest ---
    print("\n  [Random Forest]")
    rf_crop = traditional_models.rf_crop.predict(X_test)
    rf_fert = traditional_models.rf_fert.predict(X_test)
    rf_water = traditional_models.rf_water.predict(X_test)

    results["rf"] = {
        "crop_acc": accuracy_score(yc_test, rf_crop),
        "crop_f1": f1_score(yc_test, rf_crop, average="weighted"),
        "fert_acc": accuracy_score(yf_test, rf_fert),
        "fert_f1": f1_score(yf_test, rf_fert, average="weighted"),
        "water_r2": r2_score(yw_test, rf_water),
        "water_rmse": np.sqrt(mean_squared_error(yw_test, rf_water)),
        "water_mae": mean_absolute_error(yw_test, rf_water),
    }
    print(f"    Crop Acc:  {results['rf']['crop_acc']:.4f}  |  F1: {results['rf']['crop_f1']:.4f}")
    print(f"    Fert Acc:  {results['rf']['fert_acc']:.4f}  |  F1: {results['rf']['fert_f1']:.4f}")
    print(f"    Water R²:  {results['rf']['water_r2']:.4f}  |  RMSE: {results['rf']['water_rmse']:.1f}")

    # --- Neural Network ---
    print("\n  [Neural Network]")
    nn_crop, nn_fert, nn_water, _ = nn_trainer.get_test_predictions(X_test)

    results["nn"] = {
        "crop_acc": accuracy_score(yc_test, nn_crop),
        "crop_f1": f1_score(yc_test, nn_crop, average="weighted"),
        "fert_acc": accuracy_score(yf_test, nn_fert),
        "fert_f1": f1_score(yf_test, nn_fert, average="weighted"),
        "water_r2": r2_score(yw_test, nn_water),
        "water_rmse": np.sqrt(mean_squared_error(yw_test, nn_water)),
        "water_mae": mean_absolute_error(yw_test, nn_water),
    }
    print(f"    Crop Acc:  {results['nn']['crop_acc']:.4f}  |  F1: {results['nn']['crop_f1']:.4f}")
    print(f"    Fert Acc:  {results['nn']['fert_acc']:.4f}  |  F1: {results['nn']['fert_f1']:.4f}")
    print(f"    Water R²:  {results['nn']['water_r2']:.4f}  |  RMSE: {results['nn']['water_rmse']:.1f}")

    # --- Ensemble ---
    print("\n  [Stacking Ensemble]")
    ens_crop, ens_fert, ens_water = ensemble.predict(
        traditional_models, nn_trainer.model, X_test
    )

    results["ensemble"] = {
        "crop_acc": accuracy_score(yc_test, ens_crop),
        "crop_f1": f1_score(yc_test, ens_crop, average="weighted"),
        "fert_acc": accuracy_score(yf_test, ens_fert),
        "fert_f1": f1_score(yf_test, ens_fert, average="weighted"),
        "water_r2": r2_score(yw_test, ens_water),
        "water_rmse": np.sqrt(mean_squared_error(yw_test, ens_water)),
        "water_mae": mean_absolute_error(yw_test, ens_water),
    }
    print(f"    Crop Acc:  {results['ensemble']['crop_acc']:.4f}  |  F1: {results['ensemble']['crop_f1']:.4f}")
    print(f"    Fert Acc:  {results['ensemble']['fert_acc']:.4f}  |  F1: {results['ensemble']['fert_f1']:.4f}")
    print(f"    Water R²:  {results['ensemble']['water_r2']:.4f}  |  RMSE: {results['ensemble']['water_rmse']:.1f}")

    # --- Best Model Summary ---
    print("\n\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║                   BEST MODEL COMPARISON                  ║")
    print("  ╠══════════════════════════════════════════════════════════╣")

    best_crop_model = max(results, key=lambda k: results[k]["crop_acc"])
    best_fert_model = max(results, key=lambda k: results[k]["fert_acc"])
    best_water_model = max(results, key=lambda k: results[k]["water_r2"])

    print(f"  ║  Best Crop Model:      {best_crop_model:<12} "
          f"(Acc={results[best_crop_model]['crop_acc']:.4f})    ║")
    print(f"  ║  Best Fertilizer Model: {best_fert_model:<11} "
          f"(Acc={results[best_fert_model]['fert_acc']:.4f})    ║")
    print(f"  ║  Best Water Model:     {best_water_model:<12} "
          f"(R²={results[best_water_model]['water_r2']:.4f})     ║")
    print("  ╚══════════════════════════════════════════════════════════╝")

    # Detailed classification report for best ensemble
    print("\n\n  === DETAILED CLASSIFICATION REPORT (Ensemble - Crop) ===\n")
    crop_names = crop_encoder.classes_
    print(classification_report(yc_test, ens_crop, target_names=crop_names, digits=3))

    timer.end_phase("Final Evaluation on Test Set")

    return results


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """
    ╔══════════════════════════════════════════════════════════════╗
    ║           AI CROP HELPER - TRAINING PIPELINE                ║
    ║                                                              ║
    ║   Just run: python train.py                                  ║
    ║   Everything is automated. Sit back and watch.               ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(main.__doc__)

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Initialize timer
    timer = TrainingTimer()
    timer.start_global()

    # ================================================================
    # PHASE 1: DATA PREPROCESSING
    # ================================================================
    timer.start_phase("Data Preprocessing")
    preprocessor = DataPreprocessor()
    data = preprocessor.run_full_pipeline()
    timer.end_phase("Data Preprocessing")

    # ================================================================
    # PHASE 2: TRADITIONAL MODELS
    # ================================================================
    timer.start_phase("XGBoost Training")
    trad_models = TraditionalModels()
    trad_models._build_xgb_models(data["n_crop_classes"], data["n_fert_classes"])
    trad_models._build_rf_models()

    xgb_timers = trad_models.train_xgboost(
        data["X_train"], data["yc_train"], data["yf_train"], data["yw_train"],
        data["X_val"], data["yc_val"], data["yf_val"], data["yw_val"]
    )
    timer.end_phase("XGBoost Training")

    timer.start_phase("Random Forest Training")
    rf_timers = trad_models.train_random_forest(
        data["X_train"], data["yc_train"], data["yf_train"], data["yw_train"],
        data["X_val"], data["yc_val"], data["yf_val"], data["yw_val"]
    )
    timer.end_phase("Random Forest Training")

    # ================================================================
    # PHASE 3: NEURAL NETWORK
    # ================================================================
    timer.start_phase("Neural Network Training")

    nn_model = CropMultiTaskNet(
        n_features=data["n_features"],
        n_crop_classes=data["n_crop_classes"],
        n_fert_classes=data["n_fert_classes"],
        config=NN_CONFIG
    )

    print_model_summary(nn_model, data["n_features"])

    nn_trainer = NNTrainer(nn_model, NN_CONFIG)
    nn_trainer.train(data["train_loader"], data["val_loader"])

    timer.end_phase("Neural Network Training")

    # ================================================================
    # PHASE 4: ENSEMBLE
    # ================================================================
    timer.start_phase("Ensemble Stacking")

    ensemble = StackingEnsemble(data["n_crop_classes"], data["n_fert_classes"])
    ensemble.train(
        trad_models, nn_model,
        data["X_train"], data["yc_train"], data["yf_train"], data["yw_train"],
        data["X_val"], data["yc_val"], data["yf_val"], data["yw_val"]
    )

    timer.end_phase("Ensemble Stacking")

    # ================================================================
    # PHASE 5: EVALUATION
    # ================================================================
    results = evaluate_all_models(data, trad_models, nn_trainer, ensemble, timer)

    # ================================================================
    # PHASE 6: SAVE EVERYTHING
    # ================================================================
    timer.start_phase("Saving Models & Artifacts")

    trad_models.save_models()
    nn_trainer.save_model()
    ensemble.save()

    # Save final results
    results_path = os.path.join(LOG_DIR, "training_results.json")
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {
            k: float(v) for k, v in metrics.items()
        }
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"  Results saved to: {results_path}")

    # Save timing
    timing_path = os.path.join(LOG_DIR, "training_timing.json")
    timer.save(timing_path)

    timer.end_phase("Saving Models & Artifacts")

    # ================================================================
    # DONE
    # ================================================================
    timer.end_global()

    print("\n  All models and artifacts saved to:")
    print(f"    Models: {MODEL_DIR}/")
    print(f"    Logs:   {LOG_DIR}/")
    print(f"\n  Files created:")
    for f in sorted(os.listdir(MODEL_DIR)):
        size = os.path.getsize(os.path.join(MODEL_DIR, f))
        print(f"    {f}: {size/1024:.1f} KB")

    return results


if __name__ == "__main__":
    results = main()
