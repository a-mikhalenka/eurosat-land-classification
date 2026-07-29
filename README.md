# EuroSAT Land Use Classification: Comparing CNN and ResNet18 Transfer Learning
Satellite image classification comparing a CNN built and trained from scratch with transfer learning using ResNet18 on the EuroSAT dataset. The project looks at how much pretrained ImageNet features actually help when the target (satellite images) is very different from natural photos.

## Overview
The goal is to classify EuroSAT satellite images into 10 categories by land type (crops, forest, highway, residential, etc.). Two modeling approaches are compared on the same data splits:
1. A CNN built and trained from scratch
2. ResNet18 pretrained on ImageNet, first with a frozen backbone, then fine-tuned

## Data
The dataset is available on [Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset).
- **Size**: 27,000 RGB images (64x64), 10 classes
- **Classes**: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- **Splits**: official train/validation/test split (18,900 / 5,400 / 2,700), roughly balanced across classes (7-11% each)

The dataset isn't included in this repo due to size. To reproduce, download it from Kaggle.

## Methods

### Data Preparation

- **EDA**: checked class balance across splits, inspected sample images and image properties
- **Custom `Dataset` class**: loads images directly from the official CSV filenames/labels
- **Normalization**: mean/std computed using the EuroSAT dataset for the baseline CNN. For ResNet, standard ImageNet mean/std were used, since its pretrained weights expect that distribution
- **Augmentation**: random rotation and horizontal/vertical flips

### Baseline CNN

A custom CNN built and trained from scratch with three convolutional blocks (32, 64, and 128 channels), using `BatchNorm` and `Dropout`.

### Transfer Learning with ResNet18

ResNet18 pretrained on ImageNet was tested in two stages:

1. **Frozen backbone** - only the new classification head was trained
2. **Fine-tuning** - the final convolutional block (`layer4`) was unfrozen and trained with a lower learning rate

### Training

- **Loss:** Cross-Entropy Loss
- **Optimizer:** AdamW
- **Weight decay:** 1e-4
- **Learning rate scheduler:** ReduceLROnPlateau
- **Early stopping:** based on validation accuracy
- **Checkpointing:** best validation model saved


## Results

The frozen ResNet is reported on the validation set because it was an intermediate experiment, while the final CNN and fine-tuned ResNet results are reported on the test set.
| Model                 | Evaluation Set | Accuracy | Macro F1 | Weighted F1 |
| --------------------- | -------------- | -------: | -------: | ----------: |
| Baseline CNN          | Test           |     0.96 |     0.96 |        0.96 |
| ResNet18 (frozen backbone)     | Validation     |    ~0.81 |        - |           - |
| ResNet18 (fine-tuned) | Test           |     0.96 |     0.96 |        0.96 |

 
## Key findings
 
- The frozen ResNet underperformed the baseline CNN (81% vs 96% accuracy), suggesting that the pretrained ImageNet features were not a good fit for satellite images without further training.
- Fine-tuning the last convolutional block closed that gap entirely, bringing ResNet18 to match the baseline CNN's performance.
- At matched accuracy, the two models still make different mistakes per class: the Baseline CNN does slightly better on Highway (0.94 vs 0.91 F1) and Forest (0.99 vs 0.98), while the fine-tuned ResNet does better on AnnualCrop (0.97 vs 0.94) and SeaLake (0.99 vs 0.98).
- The most common confusions for both models were Highway/River and PermanentCrop/HerbaceousVegetation, classes that share similar textures and colors in satellite imagery.

## Technologies used
Python, PyTorch, torchvision, scikit-learn, pandas, matplotlib, seaborn.

## How to run
- Clone repo
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) and place it under `data/EuroSAT/`
- Install dependencies: pip install -r requirements.txt
- Open and run notebooks in order
