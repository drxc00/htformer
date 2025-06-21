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

    def forward(self, x: torch.Tensor, temporal_mask: torch.Tensor = None) -> torch.Tensor:
        # print("\n--- HierarchicalTransformer Forward Pass (Detailed Debug) ---")
        # print(f"Input x shape: {x.shape}") # Expected: (batch_size, num_frames, num_joints, 3)

        x = self.embedding(x)
        # print(f"After embedding: {x.shape}") # Expected: (batch_size, num_frames, num_joints, d_model)

        x = self.spatial_encoder(x)
        # print(f"After spatial_encoder: {x.shape}") # Expected: (batch_size, num_frames, num_joints, d_model)

        x = x.mean(dim=2)
        # print(f"After mean(dim=2) (joint pooling): {x.shape}") # Expected: (batch_size, num_frames, d_model) (e.g., 16, 200, 64)

        if temporal_mask is not None:
            # print(f"Temporal mask IS provided. Mask shape: {temporal_mask.shape}") # Should be (16, 200)

            # Ensure mask is on the correct device and is float
            temporal_mask = temporal_mask.to(x.device).float()
            # print(f"Mask dtype after .float() and .to(device): {temporal_mask.dtype}")

            # Expand mask for broadcasting: (batch, F, 1)
            mask_expanded = temporal_mask.unsqueeze(-1)
            # print(f"mask_expanded shape: {mask_expanded.shape}") # Expected: (16, 200, 1)

            # Apply mask: Element-wise multiply to zero out padded values
            masked_x = x * mask_expanded 
            # print(f"masked_x shape (after applying mask): {masked_x.shape}") # Expected: (16, 200, 64)

            # Sum valid (non-zero) elements along the frame dimension
            sum_masked_x = masked_x.sum(dim=1)
            # print(f"sum_masked_x shape (after sum(dim=1)): {sum_masked_x.shape}") # **THIS MUST BE (16, 64)**

            # Calculate the count of valid elements for each sequence
            valid_counts = temporal_mask.sum(dim=1, keepdim=True) + 1e-8 
            # print(f"valid_counts shape: {valid_counts.shape}") # Expected: (16, 1)
            # print(f"valid_counts (first 5): {valid_counts[:5].flatten()}") # Check actual lengths

            # Perform the weighted average
            x_pooled = sum_masked_x / valid_counts
            # print(f"Shape after final division (x_pooled): {x_pooled.shape}") # **THIS MUST BE (16, 64)**
            x = x_pooled # Assign pooled result back to x

        else:
            # print("Temporal mask is NOT provided. Falling back to simple mean.")
            x = x.mean(dim=1) 
            # print(f"After simple global pooling: {x.shape}")

        logits = self.classifier(x) 
        # print(f"Final logits shape: {logits.shape}") # Expected: (16, 3)
        # print("--- End HierarchicalTransformer Forward Pass (Detailed Debug) ---\n")
        return logits