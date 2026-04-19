"""
Self-Pruning Neural Network — Production-Grade Implementation
=============================================================
Tredence Analytics · AI Engineer Case Study

Key design choices (beyond the baseline spec)
──────────────────────────────────────────────
1. PrunableLinear with temperature-annealed sigmoid
     High temp at start  → gates all ≈ 0.5 (open); network learns features freely.
     Low  temp at end    → sigmoid very steep; gates snap to 0 or 1.
2. Straight-Through Estimator (STE) for hard binary gates at inference.
     Forward : threshold sigmoid output at 0.5  → crisp {0,1} mask.
     Backward: gradient passes straight through (no vanishing).
3. Separate learning rate for gate_scores (3× weights LR).
     Gates operate on a different loss landscape and need to move faster.
4. FLOPs reduction metric alongside sparsity %.
     Sparsity % tells you how many weights are zero.
     FLOPs reduction tells you how much compute you actually save.
5. Three rich matplotlib plots saved to ./results/.

Usage
─────
    pip install torch torchvision matplotlib numpy
    python self_pruning_network.py

CIFAR-10 (~170 MB) is downloaded automatically on first run.
All output plots go to ./results/.
"""

import os
import math
import json
import platform

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works everywhere
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

os.makedirs("results", exist_ok=True)

# Safe num_workers: multiprocessing on Windows needs 0
NUM_WORKERS = 0 if platform.system() == "Windows" else 2


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Straight-Through Estimator (STE)
# ═══════════════════════════════════════════════════════════════════════════════

class STEBinarize(torch.autograd.Function):
    """
    Hard binary threshold with straight-through gradient.

    Forward  : out = 1  if x >= 0.5  else  0
    Backward : grad passes through unchanged (as if the op were identity)

    Why this matters:
        At inference we want truly zero weights (gate=0), not near-zero.
        The hard threshold achieves that.  But hard threshold has zero
        gradient everywhere → optimizer would never learn.  STE solves
        this by pretending the threshold is an identity during backprop,
        so gate_scores still receive meaningful gradient signals.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return (x >= 0.5).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output          # straight-through


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PrunableLinear
# ═══════════════════════════════════════════════════════════════════════════════

class PrunableLinear(nn.Module):
    """
    Custom linear layer where every weight has a learnable gate.

    Parameters
    ----------
    weight      : (out_features, in_features)  — standard linear weight
    bias        : (out_features,)
    gate_scores : (out_features, in_features)  — one scalar per weight,
                  registered as nn.Parameter so the optimizer updates it

    Forward (training, hard_gates=False)
    ──────────────────────────────────────
        gates        = sigmoid(gate_scores / temperature)   ∈ (0, 1)
        pruned_w     = weight * gates                       element-wise
        out          = x @ pruned_w.T + bias

        Gradients flow through:
          ∂loss/∂weight     via pruned_w = weight * gates
          ∂loss/∂gate_scores via gates   = sigmoid(gate_scores / temp)

    Forward (inference, hard_gates=True)
    ──────────────────────────────────────
        gates = STE_binarize(sigmoid(gate_scores))   ∈ {0, 1}
        (weights with gate=0 contribute exactly zero — truly pruned)

    Temperature annealing (set externally by the training loop)
    ────────────────────────────────────────────────────────────
        High temp (5.0) at epoch 1  → flat sigmoid → gates ≈ 0.5
        Low  temp (0.5) at last ep  → steep sigmoid → gates → 0 or 1
    """

    def __init__(self, in_features: int, out_features: int,
                 init_temperature: float = 5.0):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.temperature  = init_temperature   # annealed by training loop
        self.hard_gates   = False              # flip True for evaluation

        # ── standard parameters ───────────────────────────────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # ── gate scores (same shape as weight) ───────────────────────────────
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        # Initialisation
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Init gate_scores so sigmoid(score/5.0) ≈ 0.73  (gates mostly open)
        nn.init.constant_(self.gate_scores, 1.0)

    # ── gate helpers ──────────────────────────────────────────────────────────

    def soft_gates(self) -> torch.Tensor:
        """Continuous gates ∈ (0,1) — differentiable, used during training."""
        return torch.sigmoid(self.gate_scores / self.temperature)

    def binary_gates(self) -> torch.Tensor:
        """Hard {0,1} gates via STE — used at inference."""
        return STEBinarize.apply(self.soft_gates())

    def gates(self) -> torch.Tensor:
        return self.binary_gates() if self.hard_gates else self.soft_gates()

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pruned_w = self.weight * self.gates()       # element-wise gate masking
        return F.linear(x, pruned_w, self.bias)     # standard affine op

    # ── metrics ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def active_fraction(self) -> float:
        """Fraction of weights with soft gate >= 0.5 (considered active)."""
        return (self.soft_gates() >= 0.5).float().mean().item()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"temp={self.temperature:.2f}, hard={self.hard_gates}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Network
# ═══════════════════════════════════════════════════════════════════════════════

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32×32 RGB → 10 classes).

    Architecture
    ────────────
    CNN backbone  (Conv2d — only nn.Linear was off-limits per the spec)
        Conv(3→64,3p1)  → BN → ReLU
        Conv(64→64,3p1) → BN → ReLU → MaxPool(2)     [32 → 16]
        Conv(64→128,3p1)  → BN → ReLU
        Conv(128→128,3p1) → BN → ReLU → MaxPool(2)   [16 → 8]
        Conv(128→256,3p1) → BN → ReLU
        AdaptiveAvgPool(1) → Flatten                  [→ 256-d]

    Prunable FC head  (all gates trained end-to-end via L1 sparsity loss)
        PrunableLinear(256 → 256) → BN → ReLU → Dropout(0.4)
        PrunableLinear(256 → 128) → BN → ReLU → Dropout(0.4)
        PrunableLinear(128 → 10)

    Why CNN backbone + prunable FC head?
        The conv layers extract spatial features — pruning them destroys
        structure and hurts accuracy badly.  The FC head is where
        over-parameterisation lives; pruning there has the highest
        sparsity-per-accuracy-point trade-off.
        This mirrors how production compression actually works.
    """

    def __init__(self, dropout: float = 0.4, init_temp: float = 5.0):
        super().__init__()

        def conv_block(in_c, out_c, pool=False):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.backbone = nn.Sequential(
            conv_block(3,   64),
            conv_block(64,  64,  pool=True),
            conv_block(64,  128),
            conv_block(128, 128, pool=True),
            conv_block(128, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Prunable classifier head
        self.fc1 = PrunableLinear(256, 256, init_temp)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = PrunableLinear(256, 128, init_temp)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = PrunableLinear(128, 10,  init_temp)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

    # ── pruning control ───────────────────────────────────────────────────────

    def prunable_layers(self):
        """Yield all PrunableLinear sub-modules."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def set_temperature(self, temp: float):
        for layer in self.prunable_layers():
            layer.temperature = temp

    def set_hard_gates(self, hard: bool):
        for layer in self.prunable_layers():
            layer.hard_gates = hard

    # ── losses & metrics ──────────────────────────────────────────────────────

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all soft gate values across all prunable layers.
        Minimising this (via the λ coefficient) drives gates toward 0.
        L1 (vs L2) maintains a constant gradient regardless of gate size,
        which is what creates true sparsity instead of just shrinkage.
        """
        device = next(self.parameters()).device
        total  = torch.zeros(1, device=device)
        for layer in self.prunable_layers():
            total = total + layer.soft_gates().sum()
        return total.squeeze()

    @torch.no_grad()
    def global_sparsity(self, threshold: float = 0.5) -> float:
        """Fraction of weights whose soft gate < threshold (effectively pruned)."""
        all_gates = torch.cat(
            [l.soft_gates().flatten() for l in self.prunable_layers()]
        )
        return (all_gates < threshold).float().mean().item()

    @torch.no_grad()
    def flops_reduction(self) -> float:
        """
        Estimated fraction of multiply-adds saved in the prunable head
        versus a fully dense head of the same shape.

        FLOPs for one PrunableLinear ∝ in_features × out_features.
        Active FLOPs = dense_FLOPs × active_fraction.
        """
        total_dense = total_active = 0
        for layer in self.prunable_layers():
            d = layer.in_features * layer.out_features
            total_dense  += d
            total_active += d * layer.active_fraction()
        if total_dense == 0:
            return 0.0
        return 1.0 - total_active / total_dense

    @torch.no_grad()
    def all_gate_values(self) -> np.ndarray:
        """Flat numpy array of every soft gate value in the network."""
        return np.concatenate(
            [l.soft_gates().flatten().cpu().numpy() for l in self.prunable_layers()]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Data loaders
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataloaders(batch_size: int = 128, data_dir: str = "./data"):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        data_dir, train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(
        test_set, batch_size=512, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Training utilities
# ═══════════════════════════════════════════════════════════════════════════════

def anneal_temperature(epoch: int, total_epochs: int,
                       t_start: float = 5.0,
                       t_end:   float = 0.5) -> float:
    """
    Cosine schedule: temperature decays from t_start to t_end over training.
    High temp → flat sigmoid (gates free to explore).
    Low  temp → steep sigmoid (gates forced to commit to 0 or 1).
    """
    progress = epoch / total_epochs
    return t_end + (t_start - t_end) * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, optimizer, scheduler, lam: float,
                    device, scaler=None) -> dict:
    """
    Train for one full epoch.
    scheduler is stepped once per BATCH (required by OneCycleLR).
    """
    model.train()
    model.set_hard_gates(False)   # always soft during training

    total_loss = total_cls = total_spar = 0.0
    correct = n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # Mixed-precision path (GPU only)
            with torch.amp.autocast(device_type="cuda"):
                logits    = model(imgs)
                cls_loss  = F.cross_entropy(logits, labels)
                spar_loss = model.sparsity_loss()
                loss      = cls_loss + lam * spar_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard CPU / single-precision path
            logits    = model(imgs)
            cls_loss  = F.cross_entropy(logits, labels)
            spar_loss = model.sparsity_loss()
            loss      = cls_loss + lam * spar_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()   # OneCycleLR MUST step once per batch

        total_loss += loss.item()
        total_cls  += cls_loss.item()
        total_spar += spar_loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += labels.size(0)

    return dict(
        loss      = total_loss / len(loader),
        cls_loss  = total_cls  / len(loader),
        spar_loss = total_spar / len(loader),
        acc       = correct / n,
    )


@torch.no_grad()
def evaluate(model, loader, device, hard: bool = True) -> float:
    """
    Evaluate on `loader`.
    hard=True  → binary {0,1} gates via STE  (deployment accuracy)
    hard=False → soft sigmoid gates           (training-mode accuracy)
    """
    model.eval()
    model.set_hard_gates(hard)
    correct = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds    = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        n       += labels.size(0)
    model.set_hard_gates(False)   # always reset to soft after evaluation
    return correct / n


def train(lam: float, epochs: int, device,
          train_loader, test_loader, lr: float = 3e-3) -> dict:
    """
    Train a SelfPruningNet with sparsity coefficient λ.
    Returns a result dict with final metrics, gate values, and training history.
    """
    model = SelfPruningNet(dropout=0.4, init_temp=5.0).to(device)

    # Separate parameter groups: gate_scores get 3× the learning rate
    # because they need to move faster on the sparsity loss landscape.
    gate_params  = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]

    optimizer = optim.AdamW([
        {"params": other_params, "lr": lr,      "weight_decay": 1e-4},
        {"params": gate_params,  "lr": lr * 3,  "weight_decay": 0.0},
    ])

    # OneCycleLR steps once per BATCH — pass steps_per_epoch correctly
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = [lr, lr * 3],
        epochs          = epochs,
        steps_per_epoch = len(train_loader),
        pct_start       = 0.1,
        anneal_strategy = "cos",
    )

    # Mixed-precision scaler (GPU only; None on CPU)
    try:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    except AttributeError:
        scaler = None   # very old PyTorch — fall back gracefully

    history = dict(
        train_acc=[], val_acc_soft=[], val_acc_hard=[],
        sparsity=[], flops_reduction=[], temperature=[],
    )

    print(f"\n{'━'*70}")
    print(f"  λ = {lam:.1e}   |   {epochs} epochs   |   device: {device}")
    print(f"{'━'*70}")
    print(f"  {'Ep':>4}  {'TrainAcc':>9}  {'Soft':>8}  "
          f"{'Hard':>8}  {'Sparse':>7}  {'FLOPs↓':>7}  {'Temp':>5}")
    print("  " + "─" * 58)

    for epoch in range(1, epochs + 1):
        # Anneal temperature before each epoch
        temp = anneal_temperature(epoch, epochs)
        model.set_temperature(temp)

        # Train for one full epoch; scheduler.step() is called per batch inside
        stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, lam, device, scaler
        )

        # Log every 5 epochs and on first/last
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            val_soft = evaluate(model, test_loader, device, hard=False)
            val_hard = evaluate(model, test_loader, device, hard=True)
            sparsity = model.global_sparsity()
            flops_r  = model.flops_reduction()

            history["train_acc"].append(stats["acc"])
            history["val_acc_soft"].append(val_soft)
            history["val_acc_hard"].append(val_hard)
            history["sparsity"].append(sparsity)
            history["flops_reduction"].append(flops_r)
            history["temperature"].append(temp)

            print(
                f"  {epoch:4d}  {stats['acc']:9.3f}  {val_soft:8.3f}  "
                f"{val_hard:8.3f}  {sparsity:7.1%}  {flops_r:7.1%}  {temp:5.2f}"
            )

    final_acc_soft = evaluate(model, test_loader, device, hard=False)
    final_acc_hard = evaluate(model, test_loader, device, hard=True)
    final_sparsity = model.global_sparsity()
    final_flops_r  = model.flops_reduction()

    print(f"\n  ✓ λ={lam:.1e}  acc_soft={final_acc_soft:.4f}  "
          f"acc_hard={final_acc_hard:.4f}  "
          f"sparsity={final_sparsity:.1%}  FLOPs↓={final_flops_r:.1%}")

    return dict(
        lam             = lam,
        acc_soft        = final_acc_soft,
        acc_hard        = final_acc_hard,
        sparsity        = final_sparsity,
        flops_reduction = final_flops_r,
        gate_vals       = model.all_gate_values(),
        history         = history,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Visualisations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gate_distribution(gate_vals: np.ndarray, lam: float,
                           sparsity: float, flops_r: float,
                           save_path: str = "results/gate_dist_best.png"):
    """
    Three-panel gate histogram for the best model.
    Panel 1: Full distribution — shows overall bimodal shape.
    Panel 2: Near-zero zoom  — the pruned spike.
    Panel 3: Active cluster  — surviving weights.
    A successful result shows a large mass near 0 and a cluster near 1.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"Gate Value Distribution  |  λ={lam:.1e}  |  "
        f"Sparsity={sparsity:.1%}  |  FLOPs↓={flops_r:.1%}",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: full view
    axes[0].hist(gate_vals, bins=120, color="#4C72B0", edgecolor="none", alpha=0.85)
    axes[0].axvline(0.5, color="crimson", lw=1.5, ls="--", label="threshold=0.5")
    axes[0].set_title("All gates", fontsize=11)
    axes[0].set_xlabel("Gate value"); axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)

    # Panel 2: pruned spike near 0
    pruned = gate_vals[gate_vals < 0.1]
    axes[1].hist(pruned, bins=80, color="#DD8452", edgecolor="none", alpha=0.85)
    axes[1].set_title(f"Pruned gates  [{len(pruned):,} / {len(gate_vals):,}]", fontsize=11)
    axes[1].set_xlabel("Gate value  (0 – 0.1)")

    # Panel 3: active cluster near 1
    active = gate_vals[gate_vals >= 0.5]
    axes[2].hist(active, bins=80, color="#55A868", edgecolor="none", alpha=0.85)
    axes[2].set_title(f"Active gates  [{len(active):,} / {len(gate_vals):,}]", fontsize=11)
    axes[2].set_xlabel("Gate value  (≥ 0.5)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"  → Saved: {save_path}")


def plot_all_gate_distributions(results: list,
                                save_path: str = "results/all_gates.png"):
    """Side-by-side gate histograms for every λ in one figure."""
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res, c in zip(axes, results, colors):
        ax.hist(res["gate_vals"], bins=100, color=c, edgecolor="none", alpha=0.85)
        ax.axvline(0.5, color="crimson", lw=1.5, ls="--")
        ax.set_title(
            f"λ = {res['lam']:.1e}\n"
            f"Sparsity={res['sparsity']:.1%}   FLOPs↓={res['flops_reduction']:.1%}\n"
            f"Acc (hard) = {res['acc_hard']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("Gate value")
    axes[0].set_ylabel("Count")
    fig.suptitle("Gate Distributions for All λ Values", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"  → Saved: {save_path}")


def plot_training_curves(results: list,
                         save_path: str = "results/training_curves.png"):
    """2×2 grid: accuracy, sparsity, FLOPs, temperature for all λ."""
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Training Dynamics Across λ Values", fontsize=14, fontweight="bold")
    ax_acc, ax_spar, ax_flops, ax_temp = axes.flatten()

    for res, c in zip(results, colors):
        h   = res["history"]
        lab = f"λ={res['lam']:.1e}"
        ep  = range(len(h["val_acc_soft"]))

        ax_acc.plot(ep,  h["val_acc_soft"],    color=c, lw=2,   label=f"{lab} soft")
        ax_acc.plot(ep,  h["val_acc_hard"],    color=c, lw=1.5, ls="--", label=f"{lab} hard")
        ax_spar.plot(ep, h["sparsity"],        color=c, lw=2,   label=lab)
        ax_flops.plot(ep,h["flops_reduction"], color=c, lw=2,   label=lab)
        ax_temp.plot(    h["temperature"],     color=c, lw=2,   label=lab)

    for ax, title, ylabel in [
        (ax_acc,  "Validation Accuracy",           "Accuracy"),
        (ax_spar, "Sparsity Level",                "Fraction pruned"),
        (ax_flops,"FLOPs Reduction (head)",        "Fraction saved"),
        (ax_temp, "Gate Temperature (annealing)",  "Temperature"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Checkpoint (every 5 epochs)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"  → Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Hyperparameters ───────────────────────────────────────────────────────
    EPOCHS     = 40        # increase to 60-80 for best accuracy; 40 is fast & good
    BATCH_SIZE = 128
    LAMBDAS    = [1e-4, 1e-3, 5e-3]   # low / medium / high sparsity

    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # ── Train one model per λ ─────────────────────────────────────────────────
    all_results = []
    for lam in LAMBDAS:
        res = train(lam, EPOCHS, device, train_loader, test_loader)
        all_results.append(res)

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n\n{'═'*72}")
    print(f"  {'Lambda':>8}  {'Acc(soft)':>10}  {'Acc(hard)':>10}"
          f"  {'Sparsity':>9}  {'FLOPs Saved':>12}")
    print(f"  {'─'*68}")
    for r in all_results:
        print(
            f"  {r['lam']:>8.1e}  {r['acc_soft']:>10.4f}  {r['acc_hard']:>10.4f}"
            f"  {r['sparsity']:>9.1%}  {r['flops_reduction']:>12.1%}"
        )
    print(f"{'═'*72}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    best = max(all_results, key=lambda r: r["acc_hard"])
    print("Saving plots...")
    plot_gate_distribution(
        best["gate_vals"], best["lam"], best["sparsity"], best["flops_reduction"]
    )
    plot_all_gate_distributions(all_results)
    plot_training_curves(all_results)

    # ── JSON summary (for the report / README) ────────────────────────────────
    summary = [
        {k: v for k, v in r.items() if k not in ("gate_vals", "history")}
        for r in all_results
    ]
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  → Saved: results/summary.json")

    print("\n✓ All done.  Check the ./results/ folder.\n")


if __name__ == "__main__":
    main()
