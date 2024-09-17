# ViUNet: A Vision Transformer-Based UNet for Image Segmentation

This repository contains the implementation of **ViUNet**, a UNet-like model built on the principles of Vision Transformers. The model applies the concept of dividing an image into smaller patches, allowing for effective segmentation tasks. ViUNet leverages both convolutional layers and transformer-based attention blocks for feature extraction and segmentation.

## Overview

ViUNet is inspired by the traditional UNet architecture but incorporates Vision Transformer (ViT) ideas by dividing the input images into smaller patches and processing them through attention blocks in the neck of the network. The model includes both encoder and decoder layers with skip connections for precise segmentation.

## Model Architecture

- **Encoder**: The encoder consists of three down-sampling blocks (`DownBlock`), which reduce the spatial dimensions of the input image and store intermediate feature maps for skip connections.
  
- **Neck (Transformer Blocks)**: The bottleneck section of the model incorporates multiple transformer-based attention blocks to capture long-range dependencies across the feature space.

- **Decoder**: The decoder reconstructs the original image using up-sampling blocks (`UPBlock`) and merges features from the encoder using skip connections. This results in precise segmentation.

- **Input and Output**:
  - **Input**: The input image is divided into patches and passed through the convolutional layers.
  - **Output**: The final output is a 2-channel segmentation map with a sigmoid activation applied.

## Files in the Repository

- `Config.py`: Configuration file that contains hyperparameters, patch size, and model settings.
- `Dataset.py`: Script for loading and processing datasets.
- `engine.py`: Main training and evaluation loops.
- `helper_functions.py`: Utility functions for training and evaluation.
- `Simple_Unet.py`: Implementation of a basic UNet model for comparison.
- `Test.ipynb`: Jupyter notebook for testing the trained model.
- `Train.ipynb`: Jupyter notebook for training the ViUNet model.
- `utils.py`: Miscellaneous utility functions.
- `VisionTransformer.py`: Script containing the Vision Transformer components used in the ViUNet model.
- `ViUnet.py`: The main script containing the ViUNet architecture.



## How It Works

1. **Patch Extraction**: The input image is split into patches, each processed independently in the early layers.
2. **Attention Mechanism**: In the neck, transformer attention blocks capture dependencies between the patches.
3. **Skip Connections**: Feature maps from the encoder are passed to the decoder, enabling fine-grained segmentation.
4. **Output**: The final output is reshaped and passed through convolutional layers to produce the segmented image.

## Current Performance and Limitations

While the ViUNet model incorporates the strengths of Vision Transformers and UNet, the results achieved so far are not as strong as expected. Specifically, the segmentation accuracy on [your dataset] shows room for improvement. 

There are several factors that could be contributing to these results:
- The attention mechanism in the transformer neck might not be fully optimized for this specific task.
- The current dataset might benefit from additional preprocessing or augmentation.
- Fine-tuning of hyperparameters and architecture modifications may lead to better performance.

This model is still a work in progress, and further experimentation and optimization are needed to achieve state-of-the-art results. Contributions, suggestions, and collaboration are welcome to improve the model.

