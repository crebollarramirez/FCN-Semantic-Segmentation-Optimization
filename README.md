# Semantic Segmentation with Fully Convolutional Neural Networks
---
## Table of Contents
- [Introduction](#introduction)
    - [Detailed Report](#detailed-report)
- [Technology Used](#technologies-used)
- [How to run](#how-to-run)
- [Methodology](#methodology)
  - [Baseline Model](#baseline-model)
  - [Improvements Over Baseline](#improvements-over-baseline)
    - [Learning Rate Scheduling](#learning-rate-scheduling)
    - [Data Augmentation](#data-augmentation)
    - [Class Weighting](#class-weighting)
- [Experimental Architectures](#experimental-architectures)
  - [Refined FCN Model (FCN2)](#1-refined-fcn-model-fcn2)
  - [ResNet-Based Model](#2-resnet-based-model)
  - [U-Net Architecture](#3-u-net-architecture)
- [Results](#results)
- [Discussions](#discussion--future-work)
- [Conclusion](#conclusion)
- [Contributors](#contributors)
- [References](#references)

## Introduction
This project explores semantic segmentation using convolutional neural networks (CNNs) with various architectures and techniques. We build upon foundational methods such as U-Net, Fully Convolutional Networks (FCNs), and DeepLab while incorporating improvements like data augmentation, class weighting, and transfer learning. Our goal is to enhance segmentation performance on the PASCAL VOC-2012 dataset.

### Detailed Report
You can access the comprehensive analysis by clicking here: [Detailed Report](./detailed_report.pdf)

## Technologies Used
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![GPU](https://img.shields.io/badge/GPU_Accelerated-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)

## How to run
### Prerequisites
- Python 3.8 or higher
- Dependencies found in the virtual environment

### Installation
Create a conda environment using the [environment.yml](http://_vscodecontentref_/1) file: <br>
    ```sh
    conda env create -f environment.yml
    conda activate venv
    ```

### Running the Model
1. Prepare the dataset:
    - Download the PASCAL VOC-2012 dataset using ```[python or python3] download.py```

2. Train the model (Note: It is highly recommended to use a GPU for training as it will significantly speed up the process. Training on a CPU may take a very long time):
    ```sh
    [python or python3] train.py
    ```

## Methodology

### Baseline Model
The baseline model follows an FCN structure with an encoder-decoder architecture:
- **Encoder**: Stacked convolutional layers (32, 64, 128, 256, 512 channels) with batch normalization and ReLU activations.
- **Decoder**: Transposed convolutions for upsampling, followed by a 1x1 convolution for pixel-wise classification.
- **Optimization**: SGD optimizer with momentum (0.9) and a learning rate of 0.01.
- **Loss Function**: Cross-Entropy Loss, optionally weighted for class imbalance.

### Improvements Over Baseline
#### Learning Rate Scheduling
We implemented a **CosineAnnealingLR** scheduler to gradually decrease the learning rate, improving convergence and stability.

#### Data Augmentation
We applied transformations using `torchvision.transforms` to improve generalization:
- **RandomHorizontalFlip** (p=0.5)
- **RandomRotation** (±5 degrees)
- **RandomResizedCrop** (224x224, scaling range 0.9 to 1.0)

#### Class Weighting
To address class imbalance, we used **median frequency balancing**, computing weights for each class based on relative frequency to emphasize underrepresented classes.

## Experimental Architectures

### 1. Refined FCN Model (FCN2)
- LeakyReLU activations with batch normalization.
- Xavier weight initialization.
- Improved transposed convolution layers for upsampling.

**Table 1: FCN2 Architecture Details**

| Layer    | Configuration / Dimensions | Activation / Notes |
|----------|----------------------------|---------------------|
| Conv1    | Conv2d: 3 → 64, Kernel=3, Stride=2, Padding=1, Dilation=1 | LeakyReLU |
| BN1      | BatchNorm2d: 64 | — |
| Conv2    | Conv2d: 64 → 256, Kernel=3, Stride=2, Padding=1, Dilation=1 | LeakyReLU |
| BN2      | BatchNorm2d: 256 | — |
| Conv3    | Conv2d: 256 → 1024, Kernel=3, Stride=2, Padding=1, Dilation=1 | LeakyReLU |
| BN3      | BatchNorm2d: 1024 | — |
| Deconv1  | ConvTranspose2d: 1024 → 256, Kernel=3, Stride=2, Padding=1, Dilation=1, Output Padding=1 | LeakyReLU |
| BN4      | BatchNorm2d: 256 | — |
| Deconv2  | ConvTranspose2d: 256 → 64, Kernel=3, Stride=2, Padding=1, Dilation=1, Output Padding=1 | LeakyReLU |
| BN5      | BatchNorm2d: 64 | — |
| Deconv3  | ConvTranspose2d: 64 → n class, Kernel=3, Stride=2, Padding=1, Dilation=1, Output Padding=1 | — |

### 2. ResNet-Based Model
- Utilizes a **pre-trained ResNet34 encoder** to leverage transfer learning.
- Decoder built with transposed convolutions and batch normalization.

**Table 2: ResNet-based Architecture Details**

| Layer    | Configuration / Dimensions | Activation / Notes |
|----------|----------------------------|---------------------|
| Encoder  | Pretrained ResNet34 truncated before FC and AvgPool; produces 512 feature maps of size 7 × 7 | — |
| Deconv1  | ConvTranspose2d: 512 → 256, Kernel=3, Stride=2, Padding=1, Output Padding=1 | ReLU |
| BN1      | BatchNorm2d: 256 | — |
| Deconv2  | ConvTranspose2d: 256 → 128, Kernel=3, Stride=2, Padding=1, Output Padding=1 | ReLU |
| BN2      | BatchNorm2d: 128 | — |
| Deconv3  | ConvTranspose2d: 128 → 64, Kernel=3, Stride=2, Padding=1, Output Padding=1 | ReLU |
| BN3      | BatchNorm2d: 64 | — |
| Deconv4  | ConvTranspose2d: 64 → 32, Kernel=3, Stride=2, Padding=1, Output Padding=1 | ReLU |
| BN4      | BatchNorm2d: 32 | — |
| Deconv5  | ConvTranspose2d: 32 → n class, Kernel=3, Stride=2, Padding=1, Output Padding=1 | — |

### 3. U-Net Architecture
- Encoder-decoder structure with **skip connections** to retain spatial information.
- Double convolution blocks in downsampling and upsampling layers.

**Table 3: U-Net Architecture Details**

| Layer    | Configuration / Dimensions | Activation / Notes |
|----------|----------------------------|---------------------|
| Down1    | DoubleConv: in channels → 64, using two 3×3 convs (Padding=1) | ReLU; Output size: 224 × 224 |
| Pool1    | MaxPool2d: 2×2 | Reduces to 112 × 112 |
| Down2    | DoubleConv: 64 → 128 | ReLU; Output size: 112 × 112 |
| Pool2    | MaxPool2d: 2×2 | Reduces to 56 × 56 |
| Down3    | DoubleConv: 128 → 256 | ReLU; Output size: 56 × 56 |
| Pool3    | MaxPool2d: 2×2 | Reduces to 28 × 28 |
| Down4    | DoubleConv: 256 → 512 | ReLU; Output size: 28 × 28 |
| Pool4    | MaxPool2d: 2×2 | Reduces to 14 × 14 |
| Bottleneck | DoubleConv: 512 → 1024 | ReLU; Size: 14 × 14 |
| Up4      | ConvTranspose2d: Upsample from 14 × 14 to 28 × 28 | — |
| UpConv4  | DoubleConv: Merge (Skip connection) yields 1024 → 512 | ReLU |
| Up3      | ConvTranspose2d: Upsample from 28 × 28 to 56 × 56 | — |
| UpConv3  | DoubleConv: Merge yields 512 → 256 | ReLU |
| Up2      | ConvTranspose2d: Upsample from 56 × 56 to 112 × 112 | — |
| UpConv2  | DoubleConv: Merge yields 256 → 128 | ReLU |
| Up1      | ConvTranspose2d: Upsample from 112 × 112 to 224 × 224 | — |
| UpConv1  | DoubleConv: Merge yields 128 → 64 | ReLU |
| FinalConv | Conv2d: 64 → out channels, Kernel=1 | — |

## Results
We evaluated models using **Pixel Accuracy** and **Mean IoU**:

| Model       | Pixel Accuracy | Mean IoU |
|------------|---------------|---------|
| Baseline Model | 72.8% | 0.0553 |
| (Cosine LR) Model | 72.3% | 0.0603 |
| (Data Augmentation) Model | 73.6% | 0.072 |
| (Class Weighting) Model | 69.4% | 0.084 |

- Learning rate scheduling showed modest improvements.
- Data augmentation helped improve generalization.
- Class weighting significantly enhanced segmentation performance for rare classes.

## Discussion & Future Work
- **Strengths**: Data augmentation and class weighting improved model performance on imbalanced classes.
- **Challenges**: IoU scores remain relatively low, suggesting room for further improvements.
- **Future Work**:
  - Enable class weighting in loss function.
  - Integrate dropout layers to reduce overfitting.
  - Experiment with optimizers like AdamW.
  - Implement additional data augmentation techniques.

## Conclusion
This project explored several improvements in CNN-based segmentation models, demonstrating the effectiveness of transfer learning, data augmentation, and class weighting. Future enhancements will focus on further optimizing performance and generalization capabilities.

## Contributors
[Christopher Rebollar-Ramirez](https://github.com/crebollarramirez) <br>
[Chi Zhang](https://github.com/Ayaaa99)


## References
- Ronneberger et al., 2015 - U-Net Architecture
- Chen et al., 2017 - DeepLab with Atrous Convolutions
- Lin et al., 2017 - Focal Loss for Class Imbalance
- He et al., 2016 - ResNet for Transfer Learning
- Simonyan & Zisserman, 2015 - VGG Architecture


