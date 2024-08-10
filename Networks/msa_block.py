import torch
from torch import nn


class MultiheadSelfAttentionBlock(nn.Module):
    """
    Implements a multi-head self-attention block (MSA block).

    This block is a fundamental component of the Vision Transformer (ViT) architecture,
    enabling the model to attend to different parts of the input data through multiple attention heads.

    Args:
        embedding_dim (int): Dimensionality of the input embedding vector (e.g., 768 for ViT-Base).
        num_heads (int): Number of attention heads (e.g., 12 for ViT-Base).
        attn_dropout (float): Dropout rate applied to the attention scores. Defaults to 0, as dropout is typically not used in ViT-Base.
    """

    def __init__(
        self, embedding_dim: int = 768, num_heads: int = 12, attn_dropout: float = 0
    ):
        super().__init__()

        # Layer normalization applied before the self-attention mechanism
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Multi-head self-attention layer to capture relationships across different parts of the input
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )  # Specifies that the batch dimension is the first dimension

    def forward(self, x):
        """
        Forward pass through the MSA block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embedding_dim].

        Returns:
            torch.Tensor: Output tensor after applying multi-head self-attention and layer normalization.
        """
        # Apply layer normalization
        x = self.layer_norm(x)

        # Apply multi-head self-attention (no attention weights needed, only the output)
        attn_output, _ = self.multihead_attn(
            query=x, key=x, value=x, need_weights=False
        )

        return attn_output
