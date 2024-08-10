import torch
from torch import nn

from Networks import PatchEmbedding
from Networks import TransformerEncoderBlock


class ViT(nn.Module):
    """
    Vision Transformer (ViT) implementation with ViT-Base hyperparameters by default.

    This model splits an input image into patches, embeds them, and processes the embeddings
    through a series of Transformer Encoder blocks. The output is passed through a classifier
    to produce predictions.

    Args:
        img_size (int): Size of the input image (e.g., 224 for ViT-Base).
        in_channels (int): Number of channels in the input image (e.g., 3 for RGB images).
        patch_size (int): Size of each patch the image is divided into (e.g., 16x16 for ViT-Base).
        num_transformer_layers (int): Number of Transformer Encoder blocks (e.g., 12 for ViT-Base).
        embedding_dim (int): Dimensionality of the patch embeddings (e.g., 768 for ViT-Base).
        mlp_size (int): Size of the hidden layer in the MLP block (e.g., 3072 for ViT-Base).
        num_heads (int): Number of attention heads in the MSA block (e.g., 12 for ViT-Base).
        attn_dropout (float): Dropout rate applied to the attention mechanism. Defaults to 0.
        mlp_dropout (float): Dropout rate applied in the MLP block. Defaults to 0.1.
        embedding_dropout (float): Dropout rate applied to the patch and position embeddings. Defaults to 0.1.
        num_classes (int): Number of output classes for classification. Defaults to 1000 (e.g., ImageNet).
    """

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_transformer_layers: int = 12,
        embedding_dim: int = 768,
        mlp_size: int = 3072,
        num_heads: int = 12,
        attn_dropout: float = 0,
        mlp_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        num_classes: int = 1000,
    ):
        super().__init__()

        # Ensure the image size is divisible by the patch size
        assert (
            img_size % patch_size == 0
        ), f"Image size {img_size} must be divisible by patch size {patch_size}."

        # Calculate the number of patches in the image
        self.num_patches = (img_size * img_size) // (patch_size**2)

        # Learnable class token that serves as a representation for the entire image
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Learnable position embeddings added to the patch embeddings
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embedding_dim)
        )

        # Dropout applied to the patch and position embeddings
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Patch embedding layer that converts image patches into embedding vectors
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim
        )

        # Stack of Transformer Encoder blocks
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Classification head that normalizes the output and maps it to the number of classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self, x):
        """
        Forward pass through the ViT model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, img_size, img_size].

        Returns:
            torch.Tensor: Logits for each class, shape [batch_size, num_classes].
        """
        batch_size = x.shape[0]

        # Expand the class token to match the batch size
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # Compute patch embeddings for the input image
        x = self.patch_embedding(x)

        # Concatenate the class token with the patch embeddings
        x = torch.cat((class_token, x), dim=1)

        # Add position embeddings to the patch embeddings
        x = x + self.position_embedding

        # Apply dropout to the embeddings
        x = self.embedding_dropout(x)

        # Pass the embeddings through the Transformer Encoder blocks
        x = self.transformer_encoder(x)

        # Use the class token (at index 0) for classification
        x = self.classifier(x[:, 0])

        return x
