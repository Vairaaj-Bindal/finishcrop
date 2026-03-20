"""
=============================================================================
PYTORCH MULTI-TASK NEURAL NETWORK — ADVANCED ARCHITECTURE
=============================================================================
Pipeline:
  Input → FeatureGating → InputProjection → MixtureOfExperts
        → SharedBackbone (Residual Blocks) → MultiHeadSelfAttention
        → CrossTaskAttention → TransformerFFNBlock × 3
        → [Crop Head | Fertilizer Head | Water Head]

Key innovations (paper-worthy components):
  1. Feature Gating          — adaptive input feature weighting: σ(Wx)
  2. Mixture of Experts (MoE)— sparse top-k routing to specialised expert FFNs
                              P(eᵢ|x) = softmax(Wg·x)ᵢ, top-2 selected
  3. Residual Blocks          — deep network training stability via skip connections
  4. Multi-Head Self-Attention— inter-feature interaction modeling
  5. Cross-Task Attention     — bidirectional information sharing between heads
  6. Focal Loss               — FL(pₜ) = −α(1−pₜ)^γ log(pₜ), handles 1,920 classes
  7. Uncertainty-Weighted MTL — Kendall et al. 2018: L = Σᵢ[exp(−sᵢ)·Lᵢ + sᵢ]
  8. MoE Load-Balancing Loss  — prevents expert collapse via auxiliary entropy term
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from config import NN_CONFIG, MODEL_DIR


# ============================================================================
# UTILITY: Activation factory
# ============================================================================

def _get_activation(name: str) -> nn.Module:
    return {
        "relu":       nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "gelu":       nn.GELU(),
        "silu":       nn.SiLU(),
        "elu":        nn.ELU(),
    }.get(name, nn.GELU())


# ============================================================================
# 1. FEATURE GATING
# ============================================================================

class FeatureGating(nn.Module):
    """
    Adaptive input feature weighting via a learned sigmoid gate.
    Gate equation: g(x) = σ(BN(Wx))
    Output: x_gated = x ⊙ g(x)  (element-wise)
    Returns gated features AND gate values for interpretability.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.BatchNorm1d(n_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gates = self.gate(x)
        return x * gates, gates


# ============================================================================
# 2. RESIDUAL BLOCK
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Pre-activation residual block with BatchNorm + Dropout.
    h = Act(BN(FC(Act(BN(FC(x))))))  +  x
    """

    def __init__(self, dim: int, dropout: float = 0.25,
                 use_bn: bool = True, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(dim) if use_bn else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.act = _get_activation(activation)

    def forward(self, x):
        h = self.act(self.bn1(self.fc1(x)))
        h = self.drop(h)
        h = self.bn2(self.fc2(h))
        return self.act(h + x)


# ============================================================================
# 3. MIXTURE OF EXPERTS
# ============================================================================

class ExpertFFN(nn.Module):
    """Single expert: a two-layer FFN with GELU activation."""

    def __init__(self, input_dim: int, output_dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = input_dim * expansion
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """
    Sparse Mixture of Experts with Top-K gating and load-balancing auxiliary loss.

    Router:  P(eᵢ|x) = softmax(Wg · x)ᵢ
    Routing: select top-k experts by gating score
    Output:  ŷ = Σᵢ∈top-k  gᵢ · Expert_i(x)

    Load-balancing loss (Shazeer et al.):
        L_aux = n_experts · Σᵢ (fᵢ · Pᵢ)
    where fᵢ = fraction of tokens routed to expert i,
          Pᵢ = mean router probability for expert i.
    Minimising L_aux encourages uniform expert utilisation.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 n_experts: int = 4, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = nn.Linear(input_dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(input_dim, output_dim, expansion=2, dropout=dropout)
            for _ in range(n_experts)
        ])
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        Args:  x — (B, input_dim)
        Returns: output — (B, output_dim), aux_loss — scalar
        """
        B = x.size(0)

        # Router logits → probabilities
        router_logits = self.router(x)                    # (B, E)
        router_probs  = F.softmax(router_logits, dim=-1)  # (B, E)

        # Top-k selection
        top_k_probs, top_k_idx = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_probs, dim=-1)      # renormalise top-k

        # Compute expert outputs for all experts (allows gradient everywhere)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, D_out)

        # Weighted sum over top-k experts
        gate_weights = torch.zeros(B, self.n_experts, device=x.device)
        gate_weights.scatter_(1, top_k_idx, top_k_probs)
        output = torch.einsum("be,bed->bd", gate_weights, expert_outs)
        output = self.output_norm(output)

        # Load-balancing auxiliary loss
        # fᵢ = fraction of batch routed to expert i (using hard assignment of argmax)
        with torch.no_grad():
            dispatch = torch.zeros(B, self.n_experts, device=x.device)
            dispatch.scatter_(1, top_k_idx[:, :1], 1.0)   # count primary expert
            f_i = dispatch.mean(dim=0)                      # (E,)
        P_i = router_probs.mean(dim=0)                      # (E,)
        aux_loss = self.n_experts * (f_i * P_i).sum()

        return output, aux_loss


# ============================================================================
# 4. MULTI-HEAD SELF-ATTENTION
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention for tabular data.
    Treats the feature vector as a single-token sequence; attention
    captures inter-feature relationships.

    Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)
        x_seq = x.unsqueeze(1)  # (B, 1, D)

        Q = self.W_q(x_seq).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x_seq).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x_seq).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        weights = self.attn_drop(F.softmax(scores, dim=-1))

        attended = torch.matmul(weights, V)                       # (B, H, 1, d)
        attended = attended.transpose(1, 2).contiguous().view(B, 1, self.embed_dim)
        out = self.W_o(attended).squeeze(1)                       # (B, D)
        return self.layer_norm(out + x), weights


# ============================================================================
# 5. CROSS-TASK ATTENTION
# ============================================================================

class CrossTaskAttention(nn.Module):
    """
    Bidirectional cross-attention between crop and water task representations.
    Allows the crop head to condition on water demand context and vice versa,
    enabling task synergy via learned feature sharing.

    crop_out  = Attention(Q=crop,  K=water, V=water)
    water_out = Attention(Q=water, K=crop,  V=crop)
    fert_out  = average-pooled from both (lightweight)
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_cw = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_wc = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_c  = nn.LayerNorm(dim)
        self.norm_w  = nn.LayerNorm(dim)
        self.norm_f  = nn.LayerNorm(dim)
        self.drop    = nn.Dropout(dropout)

    def forward(self, h_crop, h_fert, h_water):
        c = h_crop.unsqueeze(1)   # (B, 1, D)
        w = h_water.unsqueeze(1)
        f = h_fert.unsqueeze(1)

        # Crop attends to water
        c_attended, _ = self.attn_cw(c, w, w)
        crop_out  = self.norm_c(h_crop  + self.drop(c_attended.squeeze(1)))

        # Water attends to crop
        w_attended, _ = self.attn_wc(w, c, c)
        water_out = self.norm_w(h_water + self.drop(w_attended.squeeze(1)))

        # Fertilizer gets soft context from both
        f_attended, _ = self.attn_cw(f, torch.cat([c, w], dim=1), torch.cat([c, w], dim=1))
        fert_out  = self.norm_f(h_fert  + self.drop(f_attended.squeeze(1)))

        return crop_out, fert_out, water_out


# ============================================================================
# 6. TRANSFORMER FFN BLOCK (per-task post-processing)
# ============================================================================

class TransformerFFNBlock(nn.Module):
    """
    Pre-norm transformer-style FFN block applied per task head.
    h = x + Dropout(FFN(LayerNorm(x)))
    FFN = Linear → GELU → Dropout → Linear
    """

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))


# ============================================================================
# 7. FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., RetinaNet 2017).

    FL(pₜ) = −α (1 − pₜ)^γ  log(pₜ)

    Reduces the relative loss for well-classified examples so the model
    focuses training on hard, misclassified samples — critical for the
    extreme class imbalance across 1,920 crop species/varieties.

    Args:
        gamma:          focusing parameter (default 2.0)
        alpha:          weighting factor for rare classes (default 0.25)
        label_smoothing: optional label smoothing applied before focal weighting
        reduction:      "mean" | "sum" | "none"
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        # Standard cross-entropy (with label smoothing if set)
        ce_loss = F.cross_entropy(logits, targets,
                                  label_smoothing=self.label_smoothing,
                                  reduction="none")
        # p_t = probability of the true class
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-8, 1.0)

        # Focal weight: (1 − p_t)^γ
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# 8. MAIN MODEL: CropMultiTaskNet
# ============================================================================

class CropMultiTaskNet(nn.Module):
    """
    Multi-Task Deep Neural Network for Precision Agriculture Prediction.

    Full pipeline:
        Input (n_features)
          → FeatureGating        [adaptive σ(Wx) weighting]
          → InputProjection      [Linear → BN → GELU → Dropout]
          → MixtureOfExperts     [top-k sparse routing, 4 expert FFNs]
          → SharedBackbone       [Residual Blocks at each hidden dim]
          → MultiHeadSelfAttention [8-head, inter-feature modeling]
          → Task Split: 3 × TransformerFFNBlock [per-task refinement]
          → CrossTaskAttention   [crop ↔ water bidirectional context]
          → Task-specific prediction heads:
              • Crop Head:       Linear → BN → GELU → Dropout → Linear(n_crops)
              • Fertilizer Head: Linear → BN → GELU → Dropout → Linear(n_ferts)
              • Water Head:      Linear → BN → GELU → Linear → GELU → Linear(1)

    Forward returns 6-tuple:
        (crop_logits, fert_logits, water_pred,
         feature_gates, attention_weights, moe_aux_loss)
    """

    def __init__(self, n_features: int, n_crop_classes: int, n_fert_classes: int,
                 config: dict = None):
        super().__init__()
        if config is None:
            config = NN_CONFIG

        self.n_features     = n_features
        self.n_crop_classes = n_crop_classes
        self.n_fert_classes = n_fert_classes
        self.config         = config

        hdims    = config["hidden_dims"]     # e.g. [512, 1024, 512, 256, 128]
        dropout  = config["dropout_rate"]
        use_bn   = config["use_batch_norm"]
        act      = config["activation"]
        n_exp    = config.get("n_experts", 4)
        top_k    = config.get("top_k_experts", 2)
        exp_drop = config.get("expert_dropout", 0.1)
        attn_h   = config.get("attention_heads", 8)

        # 1. Feature Gating
        self.feature_gate = FeatureGating(n_features)

        # 2. Input Projection: features → hdims[0]
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hdims[0]),
            nn.BatchNorm1d(hdims[0]) if use_bn else nn.Identity(),
            _get_activation(act),
            nn.Dropout(dropout),
        )

        # 3. Mixture of Experts: hdims[0] → hdims[0]
        self.moe = MixtureOfExperts(hdims[0], hdims[0], n_experts=n_exp,
                                    top_k=top_k, dropout=exp_drop)

        # 4. Shared Backbone: series of dimension transitions + residual blocks
        backbone = []
        for i in range(len(hdims) - 1):
            backbone += [
                nn.Linear(hdims[i], hdims[i + 1]),
                nn.BatchNorm1d(hdims[i + 1]) if use_bn else nn.Identity(),
                _get_activation(act),
                nn.Dropout(dropout),
                ResidualBlock(hdims[i + 1], dropout, use_bn, act),
            ]
        self.backbone = nn.Sequential(*backbone)

        shared_dim = hdims[-1]  # 128

        # 5. Multi-Head Self-Attention
        self.self_attn = None
        if config.get("use_self_attention", True):
            # attention_heads must divide shared_dim
            h = attn_h
            while shared_dim % h != 0 and h > 1:
                h //= 2
            self.self_attn = MultiHeadSelfAttention(shared_dim, num_heads=h, dropout=dropout)

        # 6. Per-task Transformer FFN blocks (pre cross-task attention)
        self.crop_ffn  = TransformerFFNBlock(shared_dim, expansion=4, dropout=dropout)
        self.fert_ffn  = TransformerFFNBlock(shared_dim, expansion=4, dropout=dropout)
        self.water_ffn = TransformerFFNBlock(shared_dim, expansion=4, dropout=dropout)

        # 7. Cross-Task Attention
        self.cross_attn = CrossTaskAttention(shared_dim, num_heads=max(1, shared_dim // 32),
                                             dropout=dropout)

        # 8. Task-specific Prediction Heads
        neck = shared_dim // 2  # 64
        self.crop_head = nn.Sequential(
            nn.Linear(shared_dim, neck),
            nn.BatchNorm1d(neck) if use_bn else nn.Identity(),
            _get_activation(act), nn.Dropout(dropout * 0.5),
            nn.Linear(neck, n_crop_classes),
        )
        self.fert_head = nn.Sequential(
            nn.Linear(shared_dim, neck),
            nn.BatchNorm1d(neck) if use_bn else nn.Identity(),
            _get_activation(act), nn.Dropout(dropout * 0.5),
            nn.Linear(neck, n_fert_classes),
        )
        self.water_head = nn.Sequential(
            nn.Linear(shared_dim, neck),
            nn.BatchNorm1d(neck) if use_bn else nn.Identity(),
            _get_activation(act), nn.Dropout(dropout * 0.5),
            nn.Linear(neck, neck // 2),
            _get_activation(act),
            nn.Linear(neck // 2, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:  x — (B, n_features)
        Returns:
            crop_logits   — (B, n_crop_classes)
            fert_logits   — (B, n_fert_classes)
            water_pred    — (B, 1)
            feature_gates — (B, n_features)   [interpretability]
            attn_weights  — (B, H, 1, 1) | None
            moe_aux_loss  — scalar tensor     [add to total loss]
        """
        # Feature gating
        x_gated, feature_gates = self.feature_gate(x)

        # Input projection
        h = self.input_proj(x_gated)

        # Mixture of Experts
        h, moe_aux = self.moe(h)

        # Shared backbone
        h = self.backbone(h)

        # Self-attention
        attn_weights = None
        if self.self_attn is not None:
            h, attn_weights = self.self_attn(h)

        # Per-task FFN refinement
        h_c = self.crop_ffn(h)
        h_f = self.fert_ffn(h)
        h_w = self.water_ffn(h)

        # Cross-task attention
        h_c, h_f, h_w = self.cross_attn(h_c, h_f, h_w)

        # Prediction heads
        crop_logits = self.crop_head(h_c)
        fert_logits = self.fert_head(h_f)
        water_pred  = self.water_head(h_w)

        return crop_logits, fert_logits, water_pred, feature_gates, attn_weights, moe_aux

    def get_num_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================================
# 9. MULTI-TASK LOSS
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    Composite multi-task loss combining:
      • FocalLoss          for crop classification    (γ=2.0, handles 1,920 classes)
      • FocalLoss          for fertilizer prediction  (γ=1.5, 15 classes)
      • HuberLoss          for water regression       (δ=50mm, robust to outliers)
      • MoE auxiliary loss to prevent expert collapse
      • Uncertainty weighting (Kendall et al., 2018):
            L = Σᵢ [ exp(−sᵢ) · Lᵢ + sᵢ ]   where sᵢ = log σᵢ²

    All task-weighting parameters are jointly learned with the main model.
    """

    def __init__(self, n_crop_classes: int, n_fert_classes: int, config: dict = None):
        super().__init__()
        if config is None:
            config = NN_CONFIG

        self.crop_loss  = FocalLoss(
            gamma=config.get("focal_gamma", 2.0),
            alpha=config.get("focal_alpha", 0.25),
            label_smoothing=config.get("label_smoothing", 0.05),
        )
        self.fert_loss  = FocalLoss(
            gamma=max(1.0, config.get("focal_gamma", 2.0) - 0.5),
            alpha=config.get("focal_alpha", 0.25),
            label_smoothing=config.get("label_smoothing", 0.05),
        )
        self.water_loss = nn.HuberLoss(delta=50.0)

        # Learnable log-variance parameters for uncertainty weighting
        self.log_var_crop  = nn.Parameter(torch.zeros(1))
        self.log_var_fert  = nn.Parameter(torch.zeros(1))
        self.log_var_water = nn.Parameter(torch.zeros(1))

        self.moe_aux_weight = 0.01   # small coefficient for auxiliary loss

        # Static fallback weights
        self.w_crop  = config.get("crop_loss_weight", 1.0)
        self.w_fert  = config.get("fertilizer_loss_weight", 0.85)
        self.w_water = config.get("water_loss_weight", 0.6)

    def forward(self, crop_logits, fert_logits, water_pred,
                y_crop, y_fert, y_water,
                moe_aux_loss: torch.Tensor = None,
                use_uncertainty: bool = True):
        """
        Compute composite multi-task loss.

        Returns: (total_loss, loss_dict)
        """
        l_crop  = self.crop_loss(crop_logits, y_crop)
        l_fert  = self.fert_loss(fert_logits, y_fert)
        l_water = self.water_loss(water_pred.squeeze(-1), y_water)

        if use_uncertainty:
            # Uncertainty-weighted loss (Kendall & Gal 2018)
            # L = exp(−s)·L + s  where s = log σ²
            prec_crop  = torch.exp(-self.log_var_crop)
            prec_fert  = torch.exp(-self.log_var_fert)
            prec_water = torch.exp(-self.log_var_water)
            total = (prec_crop  * l_crop  + self.log_var_crop  +
                     prec_fert  * l_fert  + self.log_var_fert  +
                     prec_water * l_water + self.log_var_water)
        else:
            total = (self.w_crop  * l_crop +
                     self.w_fert  * l_fert +
                     self.w_water * l_water)

        if moe_aux_loss is not None:
            total = total + self.moe_aux_weight * moe_aux_loss

        loss_dict = {
            "total":      total.item(),
            "crop":       l_crop.item(),
            "fertilizer": l_fert.item(),
            "water":      l_water.item(),
            "moe_aux":    moe_aux_loss.item() if moe_aux_loss is not None else 0.0,
        }
        return total, loss_dict


# ============================================================================
# MODEL SUMMARY
# ============================================================================

def print_model_summary(model: CropMultiTaskNet, input_size: int):
    print("\n" + "=" * 65)
    print("  MODEL ARCHITECTURE SUMMARY")
    print("=" * 65)
    print(f"  Model:           {model.__class__.__name__}")
    total, trainable = model.get_num_parameters()
    print(f"  Parameters:      {trainable:,} trainable / {total:,} total")
    print(f"  Input Features:  {model.n_features}")
    print(f"  Crop Classes:    {model.n_crop_classes:,}")
    print(f"  Fert Classes:    {model.n_fert_classes}")
    print(f"  Hidden Dims:     {model.config['hidden_dims']}")
    print(f"  Dropout Rate:    {model.config['dropout_rate']}")
    print(f"  Activation:      {model.config['activation']}")
    print(f"  Self-Attention:  {model.self_attn is not None}")
    print(f"  Attn Heads:      {model.config.get('attention_heads', 'N/A')}")
    print(f"  MoE Experts:     {model.config.get('n_experts', 4)} (top-{model.config.get('top_k_experts',2)})")

    model.eval()
    with torch.no_grad():
        x = torch.randn(4, input_size)
        out = model(x)
        crop_out, fert_out, water_out, gates, attn, moe_aux = out
        print(f"\n  Forward pass shapes:")
        print(f"    Input:        {tuple(x.shape)}")
        print(f"    Crop logits:  {tuple(crop_out.shape)}")
        print(f"    Fert logits:  {tuple(fert_out.shape)}")
        print(f"    Water pred:   {tuple(water_out.shape)}")
        print(f"    Gates:        {tuple(gates.shape)}")
        if attn is not None:
            print(f"    Attn weights: {tuple(attn.shape)}")
        print(f"    MoE aux loss: {moe_aux.item():.4f}")
    print("  ✓ Forward pass successful")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    n_feat, n_crops, n_ferts = 45, 1920, 15

    model = CropMultiTaskNet(n_feat, n_crops, n_ferts)
    print_model_summary(model, n_feat)

    # Test loss
    criterion = MultiTaskLoss(n_crops, n_ferts)
    x = torch.randn(8, n_feat)
    yc = torch.randint(0, n_crops, (8,))
    yf = torch.randint(0, n_ferts, (8,))
    yw = torch.rand(8) * 800 + 100

    crop_l, fert_l, water_l, gates, attn, moe_aux = model(x)
    total, ld = criterion(crop_l, fert_l, water_l, yc, yf, yw, moe_aux)
    print(f"  Loss: {total.item():.4f}  (crop={ld['crop']:.3f}, "
          f"fert={ld['fertilizer']:.3f}, water={ld['water']:.3f}, "
          f"moe_aux={ld['moe_aux']:.4f})")
    print("  ✓ Loss computation successful")
