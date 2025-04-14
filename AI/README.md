# Chest X-ray Classification using DenseNet201

## Project Overview

This documentation details a deep learning system for classifying chest X-ray images using transfer learning with the DenseNet201 architecture. The model is designed to distinguish between normal X-rays and those showing signs of pneumonia (bacterial and viral), providing a tool that could potentially assist medical professionals in diagnosis.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements and Setup](#requirements-and-setup)
3. [Dataset](#dataset)
4. [Project Pipeline](#project-pipeline)
5. [Model Architecture](#model-architecture)
6. [Training Methodology](#training-methodology)
7. [Performance Evaluation](#performance-evaluation)
8. [Visualization and Analysis](#visualization-and-analysis)
9. [Usage Guide](#usage-guide)
10. [Conclusion and Future Work](#conclusion-and-future-work)
11. [References](#references)

## Introduction

Pneumonia is a significant global health concern, particularly in developing regions where access to expert radiologists may be limited. This project leverages deep learning and transfer learning techniques to create an automated system for detecting pneumonia from chest X-ray images. By utilizing DenseNet201, a powerful pre-trained convolutional neural network, the system achieves high accuracy while requiring less training data than building a model from scratch.

The project aims to provide:
- An accurate classification model for chest X-ray images
- A comparison of performance between transfer learning and custom CNN approaches
- Visual analysis tools to understand model decisions
- A practical implementation that could be integrated into clinical workflows

## Requirements and Setup

### Hardware Requirements
- GPU with CUDA support (recommended for efficient training)
- Minimum 8GB RAM (16GB+ recommended)
- Sufficient storage for dataset and model weights

### Software Dependencies
- Python 3.6+
- TensorFlow 2.x and Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- PIL/Pillow
- Kaggle API (for dataset download)

### Environment Setup
The project is optimized for Kaggle notebooks environment which provides:
- TPU/GPU acceleration
- Pre-installed libraries
- Easy dataset integration

To run locally, install dependencies with:
```
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow
```

## Dataset

The project utilizes the "Curated Chest X-ray Image Dataset for COVID19" which contains:

- **Classes**: Normal, Pneumonia-Bacterial, Pneumonia-Viral
- **Image Format**: Grayscale chest X-ray images in various resolutions
- **Data Distribution**: Variable number of images per class, requiring balancing

### Dataset Structure
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

## Project Pipeline

The implementation follows a structured pipeline:

1. **Data Preparation**
   - Loading and organizing image data
   - Creating dataframes with image paths and labels

2. **Data Preprocessing**
   - Resizing images to 224×224 pixels (DenseNet201 input size)
   - Converting to RGB format (3 channels)
   - Normalizing pixel values
   - Class balancing with downsampling/augmentation

3. **Data Splitting**
   - Train/validation/test split (70%/15%/15%)
   - Stratified sampling to maintain class distribution

4. **Model Architecture Design**
   - Loading pre-trained DenseNet201
   - Freezing base layers
   - Adding custom classification layers

5. **Model Training**
   - Transfer learning approach
   - Fine-tuning with carefully selected hyperparameters
   - Implementation of callbacks for optimization

6. **Evaluation and Analysis**
   - Performance metrics calculation
   - Visual analysis with confusion matrices and ROC curves
   - Sample prediction visualization

## Model Architecture

### Base Architecture: DenseNet201
DenseNet201 is chosen for its:
- Deep architecture with 201 layers
- Dense connectivity pattern
- Efficient parameter usage through feature reuse
- Strong performance on image classification tasks

### Transfer Learning Implementation
The model architecture consists of:

1. **Base Model**: Pre-trained DenseNet201 (weights from ImageNet)
   - Input shape: 224×224×3
   - All base layers initially frozen during training

2. **Custom Top Layers**:
   - Global Average Pooling
   - Dense layer (256 units, ReLU activation)
   - Batch Normalization
   - Dropout (0.5) for regularization
   - Output layer (3 units, Softmax activation)

### Key Architectural Features
- **Dense Connections**: Each layer receives feature maps from all preceding layers
- **Feature Reuse**: Promotes information flow and gradient propagation
- **Reduced Parameters**: Compared to similar-depth networks without dense connections
- **Custom Top**: Tailored specifically for the X-ray classification task

## Training Methodology

### Training Strategy
The training follows a two-phase approach:

1. **Phase 1: Feature Extraction**
   - Freeze all DenseNet201 layers
   - Train only the custom top layers
   - Learn task-specific features while preserving pre-trained weights

2. **Phase 2: Fine-Tuning**
   - Unfreeze the last convolutional block of DenseNet201
   - Apply a very small learning rate
   - Fine-tune both pre-trained and custom layers simultaneously

### Hyperparameters
- **Optimizer**: Adam (learning rate: 0.001 for phase 1, 0.0001 for phase 2)
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 32
- **Epochs**: Up to 30 per phase with early stopping
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### Optimization Techniques
- **Learning Rate Reduction**: ReduceLROnPlateau callback
- **Early Stopping**: To prevent overfitting
- **Class Weighting**: To handle any remaining class imbalance
- **Data Augmentation**: Applied during training (rotation, zoom, shift, flip)

## Performance Evaluation

### Metrics
The model's performance is evaluated using:

1. **Accuracy**: Overall correct classification rate
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **AUC-ROC**: Area under the Receiver Operating Characteristic curve

### Visualization Methods
- **Confusion Matrix**: Visual representation of model predictions vs. actual labels
- **ROC Curves**: Performance visualization at different classification thresholds
- **Training History**: Plots of accuracy and loss during training
- **Class Activation Maps**: Highlighting regions of importance in classification decisions

## Visualization and Analysis

### Training Progress Visualization
The project includes visualizations of:
- Training and validation accuracy across epochs
- Training and validation loss across epochs
- Learning rate adjustments

### Model Interpretation
- **Confusion Matrix Analysis**: Identifying common misclassifications
- **ROC Analysis**: Understanding classification performance for each class
- **Grad-CAM Visualization**: Highlighting image regions most influential in the model's decision
- **Sample Predictions**: Visual examples with probability distributions

### Statistical Analysis
- Class distribution before and after balancing
- Performance metrics breakdown by class
- Confidence interval estimation for metrics

## Usage Guide

### Running the Complete Pipeline
To execute the entire pipeline in Kaggle:
1. Open the notebook in Kaggle
2. Connect to a GPU accelerator
3. Run all cells sequentially

### Using the Model for Predictions
The model can be used for predictions with:

```python
def predict_xray(image_path):
    # Load and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Map to class label
    class_labels = {0: "Normal", 1: "Pneumonia-Bacterial", 2: "Pneumonia-Viral"}
    prediction_label = class_labels[predicted_class]
    confidence = predictions[0][predicted_class]
    
    return prediction_label, confidence, predictions[0]
```

### Model Deployment Options
The trained model can be:
1. **Exported**: Using `model.save('densenet_xray_classifier.h5')`
2. **Converted**: To TensorFlow Lite for mobile deployment
3. **Deployed**: Via TensorFlow Serving for API access
4. **Integrated**: Into web applications using TensorFlow.js

## Conclusion and Future Work

### Key Achievements
- Successful implementation of transfer learning for medical image classification
- High accuracy in distinguishing between normal and pneumonia X-rays
- Effective balancing of uneven class distribution
- Interpretable visualization of model decisions

### Limitations
- Limited to the three classes in the dataset
- Potential domain shift when applying to new hospital systems' X-rays
- Computational requirements of DenseNet201

### Future Improvements
- **Model Compression**: Distillation or quantization for efficiency
- **Ensemble Methods**: Combining multiple models for higher accuracy
- **Additional Classes**: Expanding to other lung conditions
- **Explainability**: More advanced techniques for model interpretation
- **Clinical Validation**: Testing in real-world clinical settings

## References

1. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700-4708).
2. Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., ... & Ng, A. Y. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv preprint arXiv:1711.05225.
3. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).
4. TensorFlow Team. Transfer learning and fine-tuning. https://www.tensorflow.org/tutorials/images/transfer_learning
5. Kaggle Dataset: "Curated Chest X-ray Image Dataset for COVID19"
