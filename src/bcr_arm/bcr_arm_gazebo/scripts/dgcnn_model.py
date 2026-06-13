#!/usr/bin/env python3
"""
DGCNN — Dynamic Graph CNN for Point Cloud Classification
=========================================================
Reference: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds",
           ACM Transactions on Graphics 2019.
           https://arxiv.org/abs/1801.07829

This implementation is architecture-compatible with the official pre-trained
ModelNet40 checkpoint from:
  https://github.com/WangYueFt/dgcnn

Checkpoint trained on ModelNet40, 40 classes, ~92.2% accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    k-Nearest Neighbours in feature space.

    Args:
        x: (B, C, N) feature tensor
        k: number of neighbours

    Returns:
        idx: (B, N, k) LongTensor of neighbour indices
    """
    # Pairwise squared distances
    inner = -2.0 * torch.bmm(x.transpose(2, 1), x)      # (B, N, N)
    xx    = (x ** 2).sum(dim=1, keepdim=True)            # (B, 1, N)
    pairwise_dist = -xx - inner - xx.transpose(2, 1)     # (B, N, N)

    idx = pairwise_dist.topk(k=k, dim=-1)[1]             # (B, N, k)
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20,
                      idx: torch.Tensor | None = None) -> torch.Tensor:
    """
    Build local neighbourhood features for EdgeConv.

    Args:
        x:   (B, C, N) input features
        k:   neighbourhood size
        idx: optional pre-computed knn indices (B, N, k)

    Returns:
        feature: (B, 2C, N, k)
    """
    B, C, N = x.size()
    device  = x.device

    if idx is None:
        idx = knn(x, k=k)

    # Gather neighbour features
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx      = idx + idx_base                               # (B, N, k)
    idx      = idx.view(-1)

    x_flat   = x.transpose(2, 1).contiguous().view(B * N, C)   # (B*N, C)
    feature  = x_flat[idx].view(B, N, k, C)                    # (B, N, k, C)

    x_centre = x.transpose(2, 1).unsqueeze(2).expand(B, N, k, C)  # (B,N,k,C)
    feature  = torch.cat([feature - x_centre, x_centre], dim=3)   # (B,N,k,2C)
    feature  = feature.permute(0, 3, 1, 2).contiguous()           # (B,2C,N,k)
    return feature


# ---------------------------------------------------------------------------
# DGCNN Model
# ---------------------------------------------------------------------------

class DGCNN(nn.Module):
    """
    DGCNN classification network for point clouds.

    Input shape:  (B, 3, N)   — batch of N 3-D points
    Output shape: (B, num_classes) — unnormalised log-probabilities

    This architecture exactly matches the official checkpoint trained on
    ModelNet40 with k=20 neighbours and an output embedding of 1024 dims.
    """

    def __init__(self, num_classes: int = 40, k: int = 20,
                 emb_dims: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.k        = k
        self.emb_dims = emb_dims

        # EdgeConv blocks
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6,   64,  kernel_size=1, bias=False), self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64,  kernel_size=1, bias=False), self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 128, kernel_size=1, bias=False), self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=1, bias=False), self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(64+64+128+256, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Classifier head
        self.linear1   = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6       = nn.BatchNorm1d(512)
        self.dp1       = nn.Dropout(p=dropout)
        self.linear2   = nn.Linear(512, 256)
        self.bn7       = nn.BatchNorm1d(256)
        self.dp2       = nn.Dropout(p=dropout)
        self.linear3   = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 3, N)
        B = x.size(0)

        # -- EdgeConv block 1 -----------------------------------------------
        x1 = get_graph_feature(x, k=self.k)          # (B, 6,   N, k)
        x1 = self.conv1(x1)                           # (B, 64,  N, k)
        x1 = x1.max(dim=-1, keepdim=False)[0]         # (B, 64,  N)

        # -- EdgeConv block 2 -----------------------------------------------
        x2 = get_graph_feature(x1, k=self.k)          # (B, 128, N, k)
        x2 = self.conv2(x2)                           # (B, 64,  N, k)
        x2 = x2.max(dim=-1, keepdim=False)[0]         # (B, 64,  N)

        # -- EdgeConv block 3 -----------------------------------------------
        x3 = get_graph_feature(x2, k=self.k)          # (B, 128, N, k)
        x3 = self.conv3(x3)                           # (B, 128, N, k)
        x3 = x3.max(dim=-1, keepdim=False)[0]         # (B, 128, N)

        # -- EdgeConv block 4 -----------------------------------------------
        x4 = get_graph_feature(x3, k=self.k)          # (B, 256, N, k)
        x4 = self.conv4(x4)                           # (B, 256, N, k)
        x4 = x4.max(dim=-1, keepdim=False)[0]         # (B, 256, N)

        # -- Aggregation ----------------------------------------------------
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)   # (B, 512, N)
        x5    = self.conv5(x_cat)                     # (B, emb, N)

        x_max = x5.max(dim=-1, keepdim=False)[0]      # (B, emb)
        x_avg = x5.mean(dim=-1)                       # (B, emb)
        x_out = torch.cat([x_max, x_avg], dim=1)      # (B, 2*emb)

        # -- MLP classifier -------------------------------------------------
        x_out = F.leaky_relu(self.bn6(self.linear1(x_out)), negative_slope=0.2)
        x_out = self.dp1(x_out)
        x_out = F.leaky_relu(self.bn7(self.linear2(x_out)), negative_slope=0.2)
        x_out = self.dp2(x_out)
        x_out = self.linear3(x_out)                   # (B, num_classes)
        return x_out
