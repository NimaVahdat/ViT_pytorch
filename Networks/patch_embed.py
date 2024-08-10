import torch
import torchvision
from torch import nn
from torchvision import transforms


class PatchEmbedding(nn.Module):
    """
    Converts a 2D input image into a 1D sequence of learnable embedding vectors.

    Args:
        in_channels (int): Number of color channels in the input image (e.g., 3 for RGB). Defaults to 3.
        patch_size (int): Size of each patch that the input image is divided into. Defaults to 16.
        embedding_dim (int): Dimensionality of the embedding vector for each patch. Defaults to 768.
    """

    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768
    ):
        super().__init__()

        self.patch_size = patch_size

        # Layer to extract patches from the input image. This is done using a convolutional layer
        # where the kernel size and stride are equal to the patch size.
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Layer to flatten the 2D patch feature maps into 1D vectors
        self.flatten = nn.Flatten(
            start_dim=2,  # flatten only the spatial dimensions (height, width)
            end_dim=3,
        )

    def forward(self, x):
        """
        Forward pass that processes the input image into a sequence of patch embeddings.

        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, in_channels, height, width].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_patches, embedding_dim], where
                          num_patches = (height // patch_size) * (width // patch_size).
        """

        # Ensure the input image size is divisible by the patch size
        image_resolution = x.shape[-1]
        assert (
            image_resolution % self.patch_size == 0
        ), f"Input image size must be divisible by patch size, got image size: {image_resolution}, patch size: {self.patch_size}"

        # Extract patches from the input image
        x_patched = self.patcher(x)

        # Flatten the patch feature maps into 1D vectors
        x_flattened = self.flatten(x_patched)

        # Rearrange the dimensions to match the expected output shape
        # Output shape: [batch_size, num_patches, embedding_dim]
        return x_flattened.permute(0, 2, 1)
