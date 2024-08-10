import torch
from torch import nn


class MLPBlock(nn.Module):
    """
    Implements a layer-normalized multilayer perceptron (MLP) block.

    This block is commonly used in the Vision Transformer (ViT) architecture, where it serves as a
    feed-forward network applied after the self-attention mechanism.

    Args:
        embedding_dim (int): Dimensionality of the input embedding vector (e.g., 768 for ViT-Base).
        mlp_size (int): Size of the hidden layer in the MLP (e.g., 3072 for ViT-Base).
        dropout (float): Dropout rate applied after each dense layer to prevent overfitting. Defaults to 0.1.
    """

    def __init__(
        self, embedding_dim: int = 768, mlp_size: int = 3072, dropout: float = 0.1
    ):
        super().__init__()

        # Layer normalization applied before the MLP layers
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Multilayer Perceptron (MLP) consisting of two linear layers with a GELU activation and dropout in between
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim, out_features=mlp_size
            ),  # First linear layer
            nn.GELU(),  # GELU activation function for non-linearity
            nn.Dropout(p=dropout),  # Dropout to prevent overfitting
            nn.Linear(
                in_features=mlp_size, out_features=embedding_dim
            ),  # Second linear layer to project back to embedding_dim
            nn.Dropout(
                p=dropout
            ),  # Another dropout layer after the second linear layer
        )

    def forward(self, x):
        """
        Forward pass through the MLP block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embedding_dim].

        Returns:
            torch.Tensor: Output tensor after applying layer normalization and the MLP layers.
        """
        # Apply layer normalization
        x = self.layer_norm(x)

        # Apply the MLP layers
        x = self.mlp(x)

        return x
