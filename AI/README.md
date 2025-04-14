# Pneumonia Detection using Chest X-ray Images

## Overview

This documentation details a machine learning system designed to classify chest X-ray images into three categories: Normal, Bacterial Pneumonia, and Viral Pneumonia. The system uses a custom Convolutional Neural Network (CNN) architecture to automatically detect and classify pneumonia from radiographic images.

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Dataset](#dataset)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Model Architecture](#model-architecture)
6. [Training Procedure](#training-procedure)
7. [Evaluation Methods](#evaluation-methods)
8. [Visualization Tools](#visualization-tools)
9. [Usage Guide](#usage-guide)
10. [Sample Results](#sample-results)
11. [Troubleshooting](#troubleshooting)

## Introduction

Pneumonia is a significant health concern worldwide, and early detection is crucial for effective treatment. This system leverages deep learning to provide automated classification of chest X-rays, potentially assisting radiologists in diagnosis. The code implements a complete pipeline from data preparation to model evaluation and prediction.

## System Requirements

### Hardware
- CPU: Multi-core processor (the code utilizes parallel processing)
- GPU: CUDA-compatible GPU recommended for training
- RAM: 8GB minimum, 16GB or more recommended

### Software Dependencies
- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow (PIL)
- Scikit-learn
- Concurrent.futures
- Multiprocessing

## Dataset

The system uses a curated dataset of chest X-ray images divided into three classes:
1. Normal (healthy lungs)
2. Pneumonia-Bacterial (bacterial pneumonia)
3. Pneumonia-Viral (viral pneumonia)

Expected directory structure:
```
/Curated X-Ray Dataset/
├── Normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Pneumonia-Bacterial/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Pneumonia-Viral/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Data Processing Pipeline

### Image Loading and Preprocessing
1. **Image Collection**: Scans directories to create a DataFrame containing image paths and labels
2. **Resizing**: All images are resized to 128×128 pixels
3. **Grayscale Conversion**: Images are converted to grayscale to reduce complexity
4. **Normalization**: Pixel values are normalized to the range [0,1]

### Data Balancing
The code implements class balancing with:
1. **Downsampling**: Classes with excessive samples are reduced to a maximum of 1600 images
2. **Data Augmentation**: Classes with insufficient samples are augmented using techniques including:
   - Random rotation (±15°)
   - Width/height shifts (±10%)
   - Shear transformation (±10%)
   - Zoom (±10%)
   - Horizontal flipping

### Dataset Splitting
The data is divided into:
- Training set (68% of data)
- Validation set (17% of data)
- Test set (15% of data)

The split is stratified to maintain class distribution across sets.

## Model Architecture

The system implements a custom CNN with the following architecture:

1. **Input Layer**: 128×128×1 (grayscale image)

2. **First Convolutional Block**:
   - Conv2D: 32 filters, 3×3 kernel, ReLU, same padding
   - Batch Normalization
   - Conv2D: 32 filters, 3×3 kernel, ReLU, same padding
   - Batch Normalization
   - MaxPooling: 2×2
   - Dropout: 10%

3. **Second Convolutional Block**:
   - Conv2D: 64 filters, 3×3 kernel, ReLU, same padding
   - Batch Normalization
   - Conv2D: 64 filters, 3×3 kernel, ReLU, same padding
   - Batch Normalization
   - MaxPooling: 2×2
   - Dropout: 25%

4. **Third Convolutional Block**:
   - Conv2D: 128 filters, 3×3 kernel, ReLU, same padding
   - Batch Normalization
   - Conv2D: 128 filters, 3×3 kernel, ReLU, same padding
   - Batch Normalization
   - MaxPooling: 2×2
   - Dropout: 10%

5. **Classifier Section**:
   - Flatten
   - Dense: 256 units, ReLU
   - Batch Normalization
   - Dropout: 20%
   - Dense: 3 units (output classes), Softmax

## Training Procedure

### Optimizer and Loss Function
- **Optimizer**: Adam with an initial learning rate of 0.001
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy

### Training Parameters
- **Batch Size**: 32
- **Maximum Epochs**: 30
- **Early Stopping**: Monitors validation loss with a patience of 10 epochs
- **Learning Rate Reduction**: Reduces learning rate by a factor of 0.5 when validation accuracy plateaus

## Evaluation Methods

The model is evaluated using multiple metrics:

1. **Accuracy**: Overall classification accuracy on the test set
2. **Confusion Matrix**: Visualization of predicted vs. actual class labels
3. **Classification Report**: Precision, recall, and F1-score for each class
4. **ROC Curves**: Receiver Operating Characteristic curves for each class
5. **AUC**: Area Under the ROC Curve for each class and average across classes

## Visualization Tools

The code includes several visualization functions:

1. **plot_training_history**: Displays accuracy and loss curves during training
2. **plot_confusion_matrix**: Visualizes the confusion matrix of predictions
3. **plot_data_distribution**: Shows class distribution in the dataset
4. **plot_sample_images**: Displays sample images from each class
5. **plot_roc_curves**: Plots ROC curves for each class

## Usage Guide

### Running the Complete Pipeline

1. Ensure the dataset is in the expected directory structure
2. Run the main script:
   ```python
   python pneumonia_detection.py
   ```

### Using the Trained Model for Predictions

The `predict_image` function demonstrates how to use the trained model for classifying new images:

```python
def predict_image(image_path):
    # Load and preprocess the image
    img = resize_image_array(image_path)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    
    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = label_map[predicted_class]
    
    return predicted_label, prediction[0][predicted_class]
```

### Customizing the Pipeline

1. **Modifying the Model Architecture**:
   - Edit the `create_and_train_custom_model` function to change layers, filters, or activation functions

2. **Adjusting Data Augmentation**:
   - Modify the parameters in the `ImageDataGenerator` initialization to change augmentation intensity

3. **Changing Preprocessing Parameters**:
   - Adjust image dimensions in the `resize_image_array` function
   - Modify normalization in the data preparation section

## Sample Results

The code outputs several visualizations:
- Training and validation accuracy/loss curves
- Confusion matrix of predictions
- ROC curves for each class
- Prediction probability distribution for sample images

It also provides textual reports:
- Dataset summary statistics
- Classification report with precision, recall, and F1-score
- Test accuracy and loss values
- Average AUC across classes

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**:
   - Reduce batch size
   - Reduce image dimensions
   - Enable memory growth with `tf.config.experimental.set_memory_growth(gpu, True)`

2. **Class Imbalance Issues**:
   - Adjust `max_images_per_class` parameter
   - Modify augmentation parameters

3. **Overfitting**:
   - Increase dropout rates
   - Add more data augmentation
   - Reduce model complexity

4. **Slow Processing**:
   - Check `max_workers` value matches available CPU cores
   - Use smaller image dimensions
   - Reduce the number of augmented images

5. **Model Saving Errors**:
   - Ensure write permissions to destination directory
   - Try saving in HDF5 format or as SavedModel format

### Handling Missing Classes

If a class is missing from the dataset, the code will print a warning:
```
Warning: No images found for class [class_name]
```

Ensure all three classes (Normal, Pneumonia-Bacterial, Pneumonia-Viral) are present in the dataset directory.
