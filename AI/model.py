import time
import os
import pathlib
from typing import List, Dict, Tuple
import warnings
import logging

# Data handling
import pandas as pd
from sklearn.model_selection import train_test_split

# Deep learning
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import backend as K
K.clear_session()
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Enable XLA optimization
tf.config.optimizer.set_jit(True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

def load_config() -> Dict:
    """Load configuration for the pipeline."""
    return {
        "img_size": (224, 224),
        "batch_size": 128,
        "epochs": 300,
        "learning_rate": 1e-3,
        "dropout_rate": 0.009,
        "validation_split": 0.15,
        "test_split": 0.15
    }

def visualize_class_distribution(labels_df: pd.DataFrame):
    """Plot the class distribution."""
    label_counts = labels_df['Finding Labels'].explode().value_counts()
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title('Class Distribution')
    plt.xticks(rotation=90)
    plt.ylabel('Frequency')
    plt.show()

def show_sample_images(image_paths: List[str]):
    """Display a few sample images."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, img_path in zip(axes, image_paths[:5]):
        img = plt.imread(img_path)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.show()

def create_directory_paths(base_path: str, dir_numbers: List[str]) -> Dict[str, str]:
    """Create and validate directory paths."""
    return {
        num: os.path.join(base_path, f"images_{num}", "images")
        for num in dir_numbers
        if os.path.exists(os.path.join(base_path, f"images_{num}", "images"))
    }

def get_image_paths(paths: List[str]) -> List[Tuple[str, str]]:
    """Get valid image file paths."""
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    return [
        (os.path.join(path, img_name), img_name)
        for path in paths
        for img_name in os.listdir(path)
        if os.path.splitext(img_name)[1].lower() in valid_extensions
    ]

def build_model(config: Dict, num_classes: int) -> Model:
    """Build the model architecture using EfficientNet."""
    base_model = EfficientNetB0(include_top=False, weights='imagenet', 
                               input_shape=(*config["img_size"], 3))
    
    # Freeze only the first 90% of layers
    freeze_layers = int(len(base_model.layers) * 0.9)
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[freeze_layers:]:
        layer.trainable = True

    inputs = layers.Input(shape=(*config["img_size"], 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Simplified dense layers
    for units in [2048, 1024, 512]:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config["dropout_rate"])(x)

    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    
    # Use mixed precision optimizer
    optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model

def create_generators(config: Dict, train_df: pd.DataFrame, 
                     val_df: pd.DataFrame, test_df: pd.DataFrame, 
                     label_columns: List[str]):
    """Create optimized data generators."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    def flow_data(datagen, df):
        return datagen.flow_from_dataframe(
            df,
            x_col='Image_Path',
            y_col=label_columns,
            target_size=config["img_size"],
            batch_size=config["batch_size"],
            class_mode='raw',
            shuffle=True
        )

    train_gen = flow_data(train_datagen, train_df)
    val_gen = flow_data(val_datagen, val_df)
    test_gen = flow_data(val_datagen, test_df)
    
    return train_gen, val_gen, test_gen

def train_model(model: Model, train_gen, val_gen, config: Dict):
    """Train the model with optimized callbacks."""
    callbacks = [
        ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Calculate steps per epoch based on dataset size and batch size
    steps_per_epoch = len(train_gen.filenames) // config["batch_size"]
    validation_steps = len(val_gen.filenames) // config["batch_size"]
    
    return model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

def main():
    logging.info('Pipeline started.')
    start_time = time.time()
    print("\n............................................\n")
    print("Loading configuration...")
    config = load_config()
    print("Configuration loaded.")

    BASE_PATH = '/kaggle/input/data'
    LABELS_PATH = os.path.join(BASE_PATH, 'Data_Entry_2017.csv')
    DIR_NUMBERS = [f"{i:03d}" for i in range(1, 13)]
    print("\n............................................\n")
    print("Reading labels file...")
    labels_df = pd.read_csv(LABELS_PATH)
    print("Labels file loaded.")
    
    print("\n............................................\n")

    print("Creating directory paths...")
    dir_paths = create_directory_paths(BASE_PATH, DIR_NUMBERS)
    print(f"Directory paths created: {len(dir_paths)} directories found.")
    print("\n............................................\n")

    print("Extracting image paths...")
    image_data = get_image_paths(list(dir_paths.values()))
    print(f"Image paths extracted: {len(image_data)} images found.")
    print("\n............................................\n")

    print("Creating DataFrame for images...")
    image_df = pd.DataFrame(image_data, columns=['Image_Path', 'Image_Name'])
    print("Image DataFrame created.")
    print("\n............................................\n")

    print("Processing labels...")
    processed_df = process_labels(image_df, labels_df)
    label_columns = [col for col in processed_df.columns if col not in ['Image_Path', 'Image_Name', 'Finding Labels']]
    print(f"Labels processed. Total classes: {len(label_columns)}.")
    print("\n............................................\n")

    print("Displaying sample images...")
    show_sample_images([img[0] for img in image_data])
    print("\n............................................\n")

    print("Splitting data into train, validation, and test sets...")
    train_df, temp_df = train_test_split(processed_df, train_size=0.7, random_state=42)
    val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=42)
    print("Data split completed.")
    print("\n............................................\n")

    print("Building model...")
    model = build_model(config, len(label_columns))
    print("Model built.")
    print("\n............................................\n")

    print("Creating data generators...")
    train_gen, val_gen, test_gen = create_generators(config, train_df, val_df, test_df, label_columns)
    print("Data generators created.")

    print("Training model...")
    history = train_model(model, train_gen, val_gen, config)
    print("Model training completed.")
    print("\n............................................\n")

    end_time = time.time()
    logging.info(f'Pipeline completed successfully in {end_time - start_time:.2f} seconds.')
    print(f"Pipeline completed successfully in {end_time - start_time:.2f} seconds.")
    print("\n............................................\n")

if __name__ == "__main__":
    main()
