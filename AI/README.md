# AI-PoweredPneumoniaDiagnosis

This project is a Python-based pipeline for classifying medical images using deep learning techniques. The pipeline includes data processing, model building, and training components. Below is a detailed documentation of the project, including setup instructions, usage, and code explanation.

## Table of Contents
1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Code Explanation](#code-explanation)
    - [DataProcessor Class](#dataprocessor-class)
    - [ModelBuilder Class](#modelbuilder-class)
    - [Trainer Class](#trainer-class)
    - [Main Function](#main-function)
5. [Dependencies](#dependencies)
6. [Contributing](#contributing)
7. [License](#license)

## Project Description
The Medical Image Classification Pipeline is designed to classify medical images into various categories based on their findings. The pipeline uses a pre-trained VGG19 model as the base model and fine-tunes it for the specific classification task. It includes data handling, model building, and training components.

## Installation
To run this project, you need to have Python installed on your system. Follow these steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/medical-image-classification.git
   cd medical-image-classification
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To use the pipeline, follow these steps:

1. Place your medical image dataset in the `data` directory. Ensure that the dataset includes a CSV file with image labels.
2. Update the `BASE_PATH` and `LABELS_PATH` variables in the `main()` function to point to your dataset and labels file.
3. Run the script:
   ```bash
   python main.py
   ```
4. The script will process the data, build the model, and train it. The best model will be saved as `best_model.h5`.

## Code Explanation
### DataProcessor Class
The `DataProcessor` class is responsible for handling data preprocessing tasks such as creating directory paths, getting image paths, and processing labels from a CSV file. It ensures that only valid image files are considered and merges labels with the image data.

### ModelBuilder Class
The `ModelBuilder` class builds the deep learning model using a pre-trained VGG19 model as the base and adds dense layers with regularization. The model is compiled with an Adam optimizer and binary cross-entropy loss for multi-label classification tasks.

### Trainer Class
The `Trainer` class handles the training process by creating data generators for training, validation, and testing sets. It also includes callbacks for model checkpointing, early stopping, and learning rate reduction on plateau. The `train()` method fits the model to the training data and validates it on the validation set.

### Main Function
The `main()` function orchestrates the entire pipeline by setting up paths, processing data, building the model, and training it. It splits the dataset into training, validation, and test sets and ensures that all necessary steps are executed in sequence.
