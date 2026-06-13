#!/usr/bin/env python3
"""
Exercice 8 - Option A : Réseau de Neurones pour l'approximation
de la cinématique directe du BCR Arm.

Pipeline :
  1. Génération d'un dataset (10 000 échantillons) via FK analytique
  2. Entraînement d'un MLP PyTorch [7 -> 64 -> 128 -> 64 -> 3]
  3. Comparaison prédictions NN vs calcul analytique
  4. Courbes d'apprentissage + analyse des erreurs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CINÉMATIQUE DIRECTE ANALYTIQUE
# ==============================================================================

L1        =  0.200
L2_OFFSET =  0.065
L3        =  0.410
L4_OFFSET = -0.065
L5        =  0.310
L6_OFFSET =  0.060
L7        =  0.105


def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])


def forward_kinematics(q):
    """Retourne la position (x, y, z) de l'effecteur pour un vecteur q."""
    dh_params = [
        (q[0],  0.025,     0,         np.pi/2),
        (q[1],  L1,        L2_OFFSET, -np.pi/2),
        (q[2],  0,         0,          np.pi/2),
        (q[3],  L3,        L4_OFFSET, -np.pi/2),
        (q[4],  0,         0,          np.pi/2),
        (q[5],  L5,        L6_OFFSET, -np.pi/2),
        (q[6],  L7,        0,          0),
    ]
    T = np.eye(4)
    for params in dh_params:
        T = T @ dh_matrix(*params)
    return T[:3, 3]


# ==============================================================================
# 2. GÉNÉRATION DU DATASET
# ==============================================================================

def generate_dataset(n_samples=10000, seed=42):
    """
    Génère n_samples paires (angles, position_effecteur).
    Les angles sont tirés uniformément dans [-pi, pi].
    """
    np.random.seed(seed)
    print(f"Génération de {n_samples} échantillons...")

    q_data  = np.random.uniform(-np.pi, np.pi, (n_samples, 7))
    xyz_data = np.array([forward_kinematics(q) for q in q_data])

    print(f"  Dataset généré : X={q_data.shape}, Y={xyz_data.shape}")
    print(f"  Position min : {xyz_data.min(axis=0)}")
    print(f"  Position max : {xyz_data.max(axis=0)}")
    return q_data.astype(np.float32), xyz_data.astype(np.float32)


# ==============================================================================
# 3. ARCHITECTURE DU RÉSEAU DE NEURONES
# ==============================================================================

class KinematicsNet(nn.Module):
    """
    MLP pour approximer la cinématique directe.
    Entrée  : 7 angles articulaires
    Sortie  : 3 coordonnées (x, y, z) de l'effecteur
    Architecture : [7 -> 64 -> 128 -> 64 -> 3]
    """
    def __init__(self):
        super(KinematicsNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.network(x)


# ==============================================================================
# 4. ENTRAÎNEMENT
# ==============================================================================

def train_model(model, train_loader, val_loader, n_epochs=100, lr=1e-3):
    """Entraîne le modèle et retourne les historiques de loss."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    train_losses, val_losses = [], []

    print(f"\nEntraînement sur {n_epochs} epochs...")
    for epoch in range(n_epochs):
        # --- Phase entraînement ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        # --- Phase validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1:3d}/{n_epochs}] "
                  f"Train Loss: {train_loss:.6f}  "
                  f"Val Loss: {val_loss:.6f}")

    return train_losses, val_losses


# ==============================================================================
# 5. ÉVALUATION ET ANALYSE DES ERREURS
# ==============================================================================

def evaluate_model(model, X_test, y_test):
    """Calcule les erreurs entre prédictions NN et valeurs analytiques."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred   = model(X_tensor).numpy()

    # Erreur euclidienne (distance 3D) en mm
    errors_3d = np.linalg.norm(y_pred - y_test, axis=1) * 1000

    # Erreur par axe en mm
    errors_x = np.abs(y_pred[:, 0] - y_test[:, 0]) * 1000
    errors_y = np.abs(y_pred[:, 1] - y_test[:, 1]) * 1000
    errors_z = np.abs(y_pred[:, 2] - y_test[:, 2]) * 1000

    print("\n=== Analyse des erreurs (en mm) ===")
    print(f"  Erreur 3D  — Moyenne: {errors_3d.mean():.3f}  "
          f"Max: {errors_3d.max():.3f}  Std: {errors_3d.std():.3f}")
    print(f"  Erreur X   — Moyenne: {errors_x.mean():.3f}  "
          f"Max: {errors_x.max():.3f}")
    print(f"  Erreur Y   — Moyenne: {errors_y.mean():.3f}  "
          f"Max: {errors_y.max():.3f}")
    print(f"  Erreur Z   — Moyenne: {errors_z.mean():.3f}  "
          f"Max: {errors_z.max():.3f}")

    return y_pred, errors_3d, errors_x, errors_y, errors_z


# ==============================================================================
# 6. VISUALISATION
# ==============================================================================

def plot_results(train_losses, val_losses,
                 y_test, y_pred,
                 errors_3d, errors_x, errors_y, errors_z):
    """Génère 4 graphiques : courbes de loss + analyse des erreurs."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BCR Arm — Réseau de Neurones pour la Cinématique Directe",
                 fontsize=14, fontweight="bold")

    # --- 1. Courbes d'apprentissage ---
    ax = axes[0, 0]
    ax.plot(train_losses, label="Train Loss", color="royalblue")
    ax.plot(val_losses,   label="Val Loss",   color="tomato")
    ax.set_title("Courbes d'apprentissage (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # --- 2. Prédictions vs Réel (axe Z) ---
    ax = axes[0, 1]
    n_show = 300
    ax.scatter(y_test[:n_show, 2], y_pred[:n_show, 2],
               alpha=0.4, s=10, color="mediumseagreen")
    z_min = min(y_test[:n_show, 2].min(), y_pred[:n_show, 2].min())
    z_max = max(y_test[:n_show, 2].max(), y_pred[:n_show, 2].max())
    ax.plot([z_min, z_max], [z_min, z_max], "r--", linewidth=1.5,
            label="Idéal")
    ax.set_title("Prédictions vs Réel — Axe Z (300 échantillons)")
    ax.set_xlabel("Z réel (m)")
    ax.set_ylabel("Z prédit (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 3. Distribution des erreurs 3D ---
    ax = axes[1, 0]
    ax.hist(errors_3d, bins=50, color="mediumpurple", edgecolor="white",
            alpha=0.85)
    ax.axvline(errors_3d.mean(), color="red", linestyle="--",
               label=f"Moyenne = {errors_3d.mean():.2f} mm")
    ax.set_title("Distribution de l'erreur euclidienne 3D")
    ax.set_xlabel("Erreur (mm)")
    ax.set_ylabel("Fréquence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 4. Erreurs par axe ---
    ax = axes[1, 1]
    ax.boxplot([errors_x, errors_y, errors_z],
               labels=["Axe X", "Axe Y", "Axe Z"],
               patch_artist=True,
               boxprops=dict(facecolor="lightskyblue"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_title("Erreurs par axe (boxplot)")
    ax.set_ylabel("Erreur absolue (mm)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kinematics_nn_results.png", dpi=150, bbox_inches="tight")
    print("\nGraphiques sauvegardés dans : kinematics_nn_results.png")
    plt.show()


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("  BCR Arm — Module IA : Cinématique Directe par NN")
    print("=" * 60)

    # --- Génération du dataset ---
    X, y = generate_dataset(n_samples=10000)

    # --- Conversion en tenseurs PyTorch ---
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    # Split : 70% train / 15% val / 15% test
    n_train = int(0.70 * len(dataset))
    n_val   = int(0.15 * len(dataset))
    n_test  = len(dataset) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256)
    test_loader  = DataLoader(test_ds,  batch_size=256)

    print(f"\nDataset : {n_train} train / {n_val} val / {n_test} test")

    # --- Modèle ---
    model = KinematicsNet()
    print(f"\nArchitecture du réseau :")
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params:,}")

    # --- Entraînement ---
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, n_epochs=100, lr=1e-3
    )

    # --- Évaluation sur le jeu de test ---
    X_test = X[-(n_test):]
    y_test = y[-(n_test):]
    y_pred, errors_3d, errors_x, errors_y, errors_z = evaluate_model(
        model, X_test, y_test
    )

    # --- Comparaison sur 5 exemples ---
    print("\n=== Comparaison sur 5 exemples ===")
    print(f"{'Angles (rad)':45s} {'Analytique (m)':30s} {'NN prédit (m)':30s} {'Erreur (mm)':>12}")
    print("-" * 120)
    for i in range(5):
        q_str  = str([f"{v:.2f}" for v in X_test[i]])
        an_str = f"({y_test[i,0]:.4f}, {y_test[i,1]:.4f}, {y_test[i,2]:.4f})"
        pr_str = f"({y_pred[i,0]:.4f}, {y_pred[i,1]:.4f}, {y_pred[i,2]:.4f})"
        err    = errors_3d[i]
        print(f"{q_str:45s} {an_str:30s} {pr_str:30s} {err:>12.3f}")

    # --- Sauvegarde du modèle ---
    torch.save(model.state_dict(), "kinematics_nn_model.pth")
    print("\nModèle sauvegardé dans : kinematics_nn_model.pth")

    # --- Visualisation ---
    plot_results(train_losses, val_losses,
                 y_test, y_pred,
                 errors_3d, errors_x, errors_y, errors_z)

    print("\n✅ Module IA terminé avec succès !")


if __name__ == "__main__":
    main()
