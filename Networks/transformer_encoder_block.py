import torch
from torch import nn

from Networks.msa_block import MultiheadSelfAttentionBlock
from Networks.mlp_block import MLPBlock


class TransformerEncoderBlock(nn.Module):
    """
    Implements a Transformer Encoder block.

    This block consists of a multi-head self-attention (MSA) layer followed by a
    multilayer perceptron (MLP) layer, both with residual connections. It is a
    fundamental component in Transformer architectures like Vision Transformer (ViT).

    Args:
        embedding_dim (int): Dimensionality of the input embedding vector (e.g., 768 for ViT-Base).
        num_heads (int): Number of attention heads in the MSA block (e.g., 12 for ViT-Base).
        mlp_size (int): Size of the hidden layer in the MLP block (e.g., 3072 for ViT-Base).
        mlp_dropout (float): Dropout rate applied in the MLP block. Defaults to 0.1.
        attn_dropout (float): Dropout rate applied in the attention layer. Defaults to 0.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: float = 0.1,
        attn_dropout: float = 0,
    ):
        super().__init__()

        # Multi-Head Self-Attention (MSA) block
        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout
        )

        # Multilayer Perceptron (MLP) block
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout
        )

    def forward(self, x):
        """
        Forward pass through the Transformer Encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embedding_dim].

        Returns:
            torch.Tensor: Output tensor after applying MSA and MLP blocks with residual connections.
        """
        # Apply the MSA block with a residual connection
        x = self.msa_block(x) + x

        # Apply the MLP block with a residual connection
        x = self.mlp_block(x) + x

        return x
