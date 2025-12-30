# Transfer Learning for Jute Pest Image Classification
## Overview
This project focuses on building and evaluating deep learning models for multi-class image classification using transfer learning. The goal is to classify images of 17 different types of jute pests by leveraging pre-trained convolutional neural networks trained on ImageNet.

Given the relatively limited size of the dataset, transfer learning is used to extract meaningful visual features while training only the final classification layers.

## Dataset
The images were obtained from https://archive.ics.uci.edu/dataset/920/jute+pest+dataset.

The dataset consists of labeled images of jute pests, divided into:
* Training set
* Validation set
* Test set

Each image belongs to one of 17 pest classes, encoded using one-hot encoding.

## Preprocessing Steps
* Images are zero-padded and resized to ensure uniform dimensions.
* Data augmentation is applied only to the training set.

**Data Augmentation**

To improve generalization and reduce overfitting, the following augmentations were applied to the training data:
* Random cropping
* Random zooming
* Random rotation
* Random flipping
* Random contrast adjustment
* Random translation

These transformations simulate real-world image variability and act as empirical regularization.

## Models and Transfer Learning

The following ImageNet pre-trained models are used as feature extractors:
* ResNet50
* ResNet101
* EfficientNetB0
* VGG16
* DenseNet201

**Transfer Learning Strategy**
* All convolutional layers of the pre-trained models are frozen.
* The original final classification layers are removed.
* The penultimate layer outputs are used as feature representations.
* A new fully connected classification head is added and trained.

**Model Architecture (Classification Head)**

Each model uses:
* Fully connected layer with L2 regularization
* Batch Normalization
* ReLU activation
* Dropout (20%)
* Final Softmax layer for multi-class prediction

**Training Configuration**
* Optimizer: Adam
* Loss Function: Categorical Cross-Entropy
* Batch Size: 5
* Epochs: 100
* Early Stopping: Based on validation loss (set to 50 to ensure at least 50 epochs)
* Model Checkpointing: Best model selected using lowest validation error

Training and validation error rates across epochs are plotted.

## Evaluation Metrics
Models are evaluated on training, validation, and test sets using the following metrics:
* Precision
* Recall
* F1 Score
* AUC (One-vs-Rest, multi-class)

These metrics provide a comprehensive view of model performance beyond accuracy.

## Results
A comparative analysis is performed across all five models to determine:
* Overall classification performance
* Generalization to unseen data
* Whether a single architecture clearly outperforms the others

Out of the 5 models, EfficientNetB0 demonstrated the strongest overall performance, achieving the highest validation and test precision, recall, AUC, and F1 scores. 
