#!/usr/bin/env python3
"""
DGCNN Object Classifier
========================
Wraps the DGCNN model to classify point clouds from ModelNet40.

The pre-trained weights are expected at:
    /home/mohamed/bcr_ws1/models/dgcnn_modelnet40.pth

If weights are not found, the classifier falls back to a randomly-
initialised model (useful for pipeline testing without GPU).

Standalone test:
    python3 classifier.py
"""

import os
import sys
import numpy as np

# Default weights path (inside the workspace)
_WS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "../../../../")  # scripts → bcr_arm_gazebo → bcr_arm → src → bcr_ws1
)
DEFAULT_WEIGHTS = os.path.join(_WS_ROOT, "models", "dgcnn_modelnet40.pth")

# ModelNet40 class labels (40 classes)
CLASS_NAMES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox",
]


class ObjectClassifier:
    """DGCNN-based 3-D point cloud classifier for ModelNet40."""

    def __init__(
        self,
        weights_path: str = DEFAULT_WEIGHTS,
        num_classes: int = 40,
        num_points: int = 1024,
        device: str | None = None,
    ):
        import torch
        from dgcnn_model import DGCNN

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device     = device
        self.num_points = num_points

        self.model = DGCNN(num_classes=num_classes).to(device)
        self.model.eval()

        weights_path = os.path.expanduser(weights_path)
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=device, weights_only=False)
            # Support both raw state-dict and checkpoint dicts
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # Strip 'module.' prefix from DataParallel checkpoints
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                print(f"[classifier] WARNING: {len(missing)} missing keys in checkpoint.")
            print(f"[classifier] Loaded weights from {weights_path}")
        else:
            print(f"[classifier] WARNING: weights not found at {weights_path}.")
            print("[classifier] Using randomly initialised model — predictions will be meaningless.")

    # ------------------------------------------------------------------
    def prepare(self, points_np: np.ndarray):
        """
        Sub-sample / pad to num_points, centre and unit-scale,
        then convert to (1, 3, N) tensor.
        """
        import torch

        n = points_np.shape[0]
        if n >= self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
        else:
            idx = np.random.choice(n, self.num_points, replace=True)
        pts = points_np[idx].astype(np.float32)

        # Normalise: centre + unit sphere
        pts -= pts.mean(axis=0)
        max_dist = np.linalg.norm(pts, axis=1).max()
        if max_dist > 1e-6:
            pts /= max_dist

        # DGCNN input: (B=1, C=3, N)
        return torch.from_numpy(pts.T).float().unsqueeze(0).to(self.device)

    def predict(self, points_np: np.ndarray):
        """
        Classify a point cloud.

        Args:
            points_np: (N, 3) numpy array of XYZ points

        Returns:
            name:  class name string (e.g. "cup")
            label: integer class index (0–39)
        """
        import torch

        with torch.no_grad():
            x      = self.prepare(points_np)
            logits = self.model(x)
            label  = int(logits.argmax(dim=1).item())

        name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"class_{label}"
        return name, label


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing ObjectClassifier with synthetic cylinder point cloud…")

    # Generate a cylinder-ish point cloud (similar to 'cup' / 'vase')
    N = 2048
    theta = np.random.uniform(0, 2 * np.pi, N)
    h     = np.random.uniform(-0.5, 0.5, N)
    r     = 0.4 + np.random.randn(N) * 0.01
    pts   = np.stack([r * np.cos(theta), r * np.sin(theta), h], axis=1).astype(np.float32)

    clf  = ObjectClassifier()
    name, label = clf.predict(pts)
    print(f"  Predicted class: {name}  (label={label})")
    print("Test complete.")
