import numpy as np
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention Is All You Need".
    Adds positional information to token embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            x + positional encoding: same shape as input
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len]

class SimpleTransformerEncoder(nn.Module):
    """
    Baseline transformer that flattens temporal and spatial dimensions
    and processes the entire sequence jointly.
    """
    def __init__(
        self,
        num_joints: int,
        num_frames: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_classes: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.seq_len = num_joints * num_frames

        self.embedding = nn.Linear(3, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_frames, num_joints, 3)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        b, f, j, _ = x.shape
        x = self.embedding(x)  # (b, f, j, d_model)
        x = x.view(b, f * j, -1)  # (b, seq_len=f*j, d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)  # Apply pos encoding (seq_len, b, d) → (b, seq_len, d)

        x = self.transformer(x)  # (b, seq_len, d_model)
        x = x.mean(dim=1)  # Global average pooling
        logits = self.classifier(x)  # (b, num_classes)
        return logits
