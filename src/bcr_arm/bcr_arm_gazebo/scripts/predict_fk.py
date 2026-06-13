#!/usr/bin/env python3
"""
Interactive Forward Kinematics Predictor — BCR Arm
Loads the trained NN model and predicts (x, y, z) from user-provided joint angles.

Usage:
    uv run python3 predict_fk.py
"""

import os
import torch
import torch.nn as nn
import numpy as np


# ==============================================================================
# Model architecture — must match kinematics_nn.py exactly
# ==============================================================================

class KinematicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 64),    nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 128),  nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64),  nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.network(x)


# ==============================================================================
# Analytical FK (for comparison)
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
    ca, sa = np.cos(alpha),  np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])

def analytical_fk(q):
    dh_params = [
        (q[0], 0.025,     0,         np.pi/2),
        (q[1], L1,        L2_OFFSET, -np.pi/2),
        (q[2], 0,         0,          np.pi/2),
        (q[3], L3,        L4_OFFSET, -np.pi/2),
        (q[4], 0,         0,          np.pi/2),
        (q[5], L5,        L6_OFFSET, -np.pi/2),
        (q[6], L7,        0,          0),
    ]
    T = np.eye(4)
    for params in dh_params:
        T = T @ dh_matrix(*params)
    return T[:3, 3]


# ==============================================================================
# Load model
# ==============================================================================

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"\n❌ Model file not found: {model_path}")
        print("   Please run kinematics_nn.py first to train and save the model.")
        exit(1)

    model = KinematicsNet()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"✅ Model loaded from: {model_path}\n")
    return model


# ==============================================================================
# Predict
# ==============================================================================

def predict(model, angles):
    x = torch.tensor([angles], dtype=torch.float32)
    with torch.no_grad():
        xyz = model(x).numpy()[0]
    return xyz


# ==============================================================================
# User input helpers
# ==============================================================================

def get_angle(joint_num):
    """Ask the user for a single joint angle, accepting degrees or radians."""
    while True:
        raw = input(f"  Joint {joint_num} : ").strip()
        if raw == "":
            print("    ⚠️  Please enter a value.")
            continue
        try:
            value = float(raw)
            return value
        except ValueError:
            print("    ⚠️  Invalid number, try again.")


def ask_unit():
    """Ask whether the user wants to enter angles in degrees or radians."""
    while True:
        choice = input("Enter angles in (r)adians or (d)egrees? [r/d]: ").strip().lower()
        if choice in ("r", "radians", ""):
            return "radians"
        elif choice in ("d", "degrees"):
            return "degrees"
        else:
            print("  ⚠️  Please type 'r' for radians or 'd' for degrees.")


# ==============================================================================
# Main loop
# ==============================================================================

def main():
    print("=" * 55)
    print("   BCR Arm — Interactive FK Predictor (Neural Network)")
    print("=" * 55)

    # Locate model file — look next to this script first, then home dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "kinematics_nn_model.pth"),
        os.path.expanduser("~/kinematics_nn_model.pth"),
        "kinematics_nn_model.pth",
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    model = load_model(model_path)

    while True:
        print("\nEnter the 7 joint angles for the BCR Arm:")

        unit = ask_unit()
        print()

        angles_input = []
        for i in range(1, 8):
            val = get_angle(i)
            angles_input.append(val)

        # Convert to radians if needed
        if unit == "degrees":
            angles_rad = [np.deg2rad(a) for a in angles_input]
        else:
            angles_rad = angles_input

        # NN prediction
        xyz_nn = predict(model, angles_rad)

        # Analytical FK for comparison
        xyz_analytical = analytical_fk(np.array(angles_rad))

        # Error
        error_mm = np.linalg.norm(xyz_nn - xyz_analytical) * 1000

        # Display results
        print("\n" + "─" * 55)
        print(f"  Input angles ({unit}):")
        for i, (a_in, a_rad) in enumerate(zip(angles_input, angles_rad), 1):
            if unit == "degrees":
                print(f"    Joint {i}: {a_in:.2f}°  ({a_rad:.4f} rad)")
            else:
                print(f"    Joint {i}: {a_rad:.4f} rad  ({np.rad2deg(a_rad):.2f}°)")

        print(f"\n  📍 NN Predicted position:")
        print(f"     x = {xyz_nn[0]:+.4f} m")
        print(f"     y = {xyz_nn[1]:+.4f} m")
        print(f"     z = {xyz_nn[2]:+.4f} m")

        print(f"\n  📐 Analytical FK position (ground truth):")
        print(f"     x = {xyz_analytical[0]:+.4f} m")
        print(f"     y = {xyz_analytical[1]:+.4f} m")
        print(f"     z = {xyz_analytical[2]:+.4f} m")

        print(f"\n  📏 NN vs Analytical error: {error_mm:.3f} mm")
        print("─" * 55)

        # Ask to continue
        again = input("\nPredict another pose? [y/n]: ").strip().lower()
        if again not in ("y", "yes", ""):
            print("\nGoodbye! 👋")
            break


if __name__ == "__main__":
    main()
