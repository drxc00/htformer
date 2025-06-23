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
    
class SpatialTransformerEncoder(nn.Module):
    """
    Captures spatial correlations between joints within each frame.
    Processes all frames in parallel for efficiency.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_joints: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_joints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_frames, num_joints, d_model)
        Returns:
            y: Tensor of same shape as input
        """
        b, f, j, d = x.shape
        # Combine batch and frame dims: treat each frame as independent sequence
        x = x.view(b * f, j, d).transpose(0, 1)  # (seq_len=j, batch=b*f, d_model=d)
        x = self.pos_encoder(x)
        y = self.transformer(x)  # (j, b*f, d)
        y = y.transpose(0, 1).view(b, f, j, d)
        return y

class TemporalTransformerEncoder(nn.Module):
    """
    Models temporal dependencies across frames.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_frames: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_frames, d_model)
        Returns:
            y: Tensor of same shape as input
        """
        # Transformer expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)  # (num_frames, batch, d_model)
        x = self.pos_encoder(x)
        y = self.transformer(x)
        y = y.transpose(0, 1)  # (batch, num_frames, d_model)
        return y
    
class HierarchicalTransformer(nn.Module):
    """
    Hierarchical Transformer combining spatial and temporal encoders for exercise recognition.
    Input: X of shape (batch_size, num_frames, num_joints, 3)
    Output: logits over exercise classes
    """
    def __init__(
        self,
        num_joints: int,
        num_frames: int,
        d_model: int,
        nhead: int,
        num_spatial_layers: int,
        num_temporal_layers: int,
        num_classes: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        # Embed 3D coordinates into d_model dims
        self.embedding = nn.Linear(3, d_model)

        # Spatial transformer to capture joint correlations per frame
        self.spatial_encoder = SpatialTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_spatial_layers,
            num_joints=num_joints,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Temporal transformer to capture motion across frames
        self.temporal_encoder = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_temporal_layers,
            num_frames=num_frames,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_frames, num_joints, 3)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Embed coordinates
        x = self.embedding(x)  # (batch, F, J, d_model)

        # Spatial encoding
        x = self.spatial_encoder(x)  # (batch, F, J, d_model)

        # Aggregate joints per frame (mean pooling)
        x = x.mean(dim=2)  # (batch, F, d_model)

        # Temporal encoding
        x = self.temporal_encoder(x)  # (batch, F, d_model)

        # Global pooling across frames
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        return logits