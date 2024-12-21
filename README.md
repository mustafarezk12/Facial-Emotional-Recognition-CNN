# Facial-Emotional-Recognition-CNN

![Facial Emotion Detection ](facial recognition.gif)


## Overview
This project aims to classify facial expressions using a Convolutional Neural Network (CNN). The model is trained on a dataset of facial images and can predict emotions such as happiness, sadness, anger, etc.

## Dataset
The dataset contains images categorized into 7 different expressions. Each image is preprocessed and resized to 150x150 pixels before being fed into the model.

## Model Architecture
The CNN model is composed of several layers:
- **Convolutional Layers**: Extract features from the image using filters.
- **Pooling Layers**: Reduce the dimensionality of the feature maps.
- **Dropout Layers**: Prevent overfitting by randomly setting a fraction of input units to 0.
- **Fully Connected Layers**: Perform classification based on extracted features.

### Detailed Architecture:
1. **Conv2D Layer**: 32 filters, kernel size (3, 3), activation 'ReLU'.
2. **MaxPooling2D Layer**: Pool size (2, 2).
3. **Dropout Layer**: 25% dropout rate.
4. **Conv2D Layer**: 64 filters, kernel size (3, 3), activation 'ReLU'.
5. **MaxPooling2D Layer**: Pool size (2, 2).
6. **Dropout Layer**: 25% dropout rate.
7. **Flatten Layer**: Flattens the 2D matrix into a 1D vector.
8. **Dense Layer**: 128 units, activation 'ReLU', L2 regularization.
9. **Dropout Layer**: 50% dropout rate.
10. **Output Layer**: Softmax activation with 7 units for 7 classes.

## How CNNs Work
Convolutional Neural Networks (CNNs) are deep learning models designed to process structured grid data, such as images. CNNs use convolutional layers that apply a set of filters to the input image, extracting features like edges, textures, and patterns.

### Image Processing:
1. **Rescaling**: Images are rescaled to values between 0 and 1.
2. **Data Augmentation**: Techniques like rotation, zoom, and flip are applied to increase data diversity and prevent overfitting.
3. **Normalization**: Input images are standardized to ensure consistent input distribution.

## Results and Evaluation
The model achieves an accuracy of 89.76% on the training data and 64.22% on the validation data. Despite overfitting, it can be improved with further tuning and data augmentation.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/mustafarezk12/Facial-Emotional-Recognition-CNN.git
