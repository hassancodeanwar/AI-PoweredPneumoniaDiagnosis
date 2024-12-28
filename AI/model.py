import os
import time
import warnings
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress warnings
warnings.filterwarnings("ignore")

@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    base_path: str
    labels_path: str
    dir_numbers: List[str]
    train_size: float = 0.7
    val_size: float = 0.2
    random_state: int = 42

class MedicalImageProcessor:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.processed_df: Optional[pd.DataFrame] = None
        self.balanced_df: Optional[pd.DataFrame] = None
        self._validate_paths()

    def _validate_paths(self) -> None:
        if not os.path.exists(self.config.base_path):
            raise FileNotFoundError(f"Base path not found: {self.config.base_path}")
        if not os.path.exists(self.config.labels_path):
            raise FileNotFoundError(f"Labels file not found: {self.config.labels_path}")

    def process_dataset(self) -> pd.DataFrame:
        dir_paths = [os.path.join(self.config.base_path, f"images_{num}", "images") for num in self.config.dir_numbers]
        image_data = [(os.path.join(p, f), f) for p in dir_paths for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
        image_df = pd.DataFrame(image_data, columns=['Image_Path', 'Image_Name'])
        labels = pd.read_csv(self.config.labels_path)
        merged_df = pd.merge(image_df, labels[['Image Index', 'Finding Labels']], left_on='Image_Name', right_on='Image Index', how='left').drop(columns=['Image Index'])
        merged_df['Finding Labels'] = merged_df['Finding Labels'].apply(lambda x: x.split('|'))
        for label in set(label for labels in merged_df['Finding Labels'] for label in labels):
            merged_df[label] = merged_df['Finding Labels'].apply(lambda x: 1 if label in x else 0)
        self.processed_df = merged_df
        return self.processed_df

    def balance_dataset(self, strategy: str = "oversample") -> pd.DataFrame:
        if self.processed_df is None:
            raise ValueError("Process dataset first.")
        label_cols = [col for col in self.processed_df.columns if col not in ['Image_Path', 'Image_Name', 'Finding Labels']]
        self.processed_df['Label_Class'] = self.processed_df[label_cols].idxmax(axis=1)

        if strategy == "oversample":
            oversampler = RandomOverSampler(random_state=self.config.random_state)
            X_res, y_res = oversampler.fit_resample(self.processed_df.drop(columns=['Label_Class']), self.processed_df['Label_Class'])
            self.balanced_df = pd.concat([X_res, y_res], axis=1)
        elif strategy == "undersample":
            min_class_size = self.processed_df['Label_Class'].value_counts().min()
            self.balanced_df = pd.concat([resample(self.processed_df[self.processed_df['Label_Class'] == cls], replace=False, n_samples=min_class_size, random_state=self.config.random_state) for cls in self.processed_df['Label_Class'].unique()])
        else:
            raise ValueError("Invalid strategy. Use 'oversample' or 'undersample'")
        return self.balanced_df

    def split_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        label_cols = [col for col in self.balanced_df.columns if col not in ['Image_Path', 'Image_Name', 'Finding Labels', 'Label_Class']]
        train, temp = train_test_split(self.balanced_df, train_size=self.config.train_size, stratify=self.balanced_df[label_cols].sum(axis=1), random_state=self.config.random_state)
        val_size_adjusted = self.config.val_size / (1 - self.config.train_size)
        val, test = train_test_split(temp, train_size=val_size_adjusted, stratify=temp[label_cols].sum(axis=1), random_state=self.config.random_state)
        return train, val, test

class ModelTrainer:
    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3), batch_size: int = 32, epochs: int = 50, learning_rate: float = 0.0001, model_save_path: str = 'models'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_save_path = model_save_path
        self.model = None
        os.makedirs(model_save_path, exist_ok=True)

    def build_model(self) -> Model:
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape)
        for layer in base_model.layers:
            layer.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='sigmoid')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
        return self.model

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, label_cols: List[str]) -> Dict:
        train_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True).flow_from_dataframe(train_df, x_col='Image_Path', y_col=label_cols, target_size=self.input_shape[:2], batch_size=self.batch_size, class_mode='raw')
        val_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input).flow_from_dataframe(val_df, x_col='Image_Path', y_col=label_cols, target_size=self.input_shape[:2], batch_size=self.batch_size, class_mode='raw')
        callbacks = [
            ModelCheckpoint(filepath=os.path.join(self.model_save_path, 'model_best.h5'), monitor='val_accuracy', save_best_only=True),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)]
        self.history = self.model.fit(train_gen, validation_data=val_gen, epochs=self.epochs, callbacks=callbacks)
        return self.history.history

    def evaluate(self, test_df: pd.DataFrame, label_cols: List[str]) -> Tuple[float, float]:
        test_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input).flow_from_dataframe(test_df, x_col='Image_Path', y_col=label_cols, target_size=self.input_shape[:2], batch_size=self.batch_size, class_mode='raw')
        return self.model.evaluate(test_gen)

def main():
    print("Initializing configuration...")
    config = DatasetConfig(base_path='/kaggle/input/data', labels_path='/kaggle/input/data/Data_Entry_2017.csv', dir_numbers=[f"{i:03d}" for i in range(1, 13)])

    print("Processing dataset...")
    processor = MedicalImageProcessor(config)
    processed_df = processor.process_dataset()
    print("Dataset processed.")

    print("Balancing dataset...")
    balanced_df = processor.balance_dataset()
    print("Dataset balanced.")

    print("Splitting dataset...")
    train_df, val_df, test_df = processor.split_dataset()
    print("Dataset split into train, validation, and test sets.")

    print("Preparing model trainer...")
    label_cols = [col for col in balanced_df.columns if col not in ['Image_Path', 'Image_Name', 'Finding Labels', 'Label_Class']]
    trainer = ModelTrainer(num_classes=len(label_cols))

    print("Building model...")
    trainer.build_model()
    print("Model built.")

    print("Training model...")
    trainer.train(train_df, val_df, label_cols)
    print("Model trained.")

    print("Evaluating model...")
    loss, accuracy = trainer.evaluate(test_df, label_cols)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
