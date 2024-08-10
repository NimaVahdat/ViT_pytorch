# Vision Transformer (ViT) - PyTorch Implementation
This repository contains a PyTorch implementation of the Vision Transformer (ViT), inspired by the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929).

## Overview
The Vision Transformer (ViT) is a groundbreaking architecture that applies transformers, typically used for natural language processing, to the domain of image recognition. By treating images as sequences of patches (analogous to words in text), ViT achieves state-of-the-art performance on various image recognition benchmarks, often surpassing convolutional neural networks (CNNs).

Key Features:
* Patch Embedding: An image is divided into fixed-size patches, which are then flattened and linearly embedded.
* Transformer Encoder: The sequence of patch embeddings is processed by a stack of transformer encoder layers.
* Class Token: A learnable embedding is prepended to the sequence, representing the entire image for classification.
* Position Embeddings: Added to patch embeddings to retain positional information.
* Fully Connected Classifier: The output corresponding to the class token is passed through a fully connected layer to produce the final classification logits.

## Implementation Details:
This implementation follows the ViT-Base hyperparameters from the paper:

* Input Image Size: 224x224
* Patch Size: 16x16
* Embedding Dimension: 768
* Number of Transformer Layers: 12
* Number of Attention Heads: 12
* MLP Size: 3072
* Dropout: Applied to embeddings, attention, and MLP layers

## Usage

### Inference
For inference on new images:

```python
from vit import ViT
import torch
from PIL import Image
from torchvision import transforms

# Load the model
model = ViT(num_classes=1000)  # Adjust number of classes as per your dataset
model.load_state_dict(torch.load('path/to/your/model.pth'))
model.eval()

# Preprocess the image
image = Image.open('path/to/image.jpg')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = preprocess(image).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(image)
    predicted_class = logits.argmax(dim=1).item()

print(f'Predicted Class: {predicted_class}')
```

## Replication of ViT Paper
This project is a faithful replication of the Vision Transformer model as described in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". The implementation closely follows the architecture and hyperparameters specified in the original paper, aiming to reproduce the results reported by the authors.

## Understanding Vision Transformers
Transformers have revolutionized the field of natural language processing by capturing long-range dependencies through self-attention mechanisms. ViT extends this paradigm to the domain of computer vision, where images are first split into patches, treated similarly to tokens (words) in NLP, and then processed using a standard transformer architecture.

## Why ViT?
* Scalability: ViT scales efficiently with dataset size and model size, achieving superior performance when trained on large datasets.
* Simplicity: The architecture is more straightforward compared to convolutional networks, as it avoids complex inductive biases inherent to CNNs.
* Flexibility: By using transformers, ViT can benefit from recent advances in the transformer-based architectures, including improvements in training techniques, optimizers, and hardware acceleration.


## License
This project is licensed under the MIT License. See the LICENSE file for details.
