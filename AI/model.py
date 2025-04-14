import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import concurrent.futures
import multiprocessing

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(e)

# Get the number of CPU cores for parallel processing
max_workers = multiprocessing.cpu_count()
print(f"Number of CPU cores: {max_workers}")

# Function to create DataFrame from image directory
def create_dataframe(data_dir):
    # Only use the three specified classes: Normal, Pneumonia-Bacterial, Pneumonia-Viral
    valid_classes = ["Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]
    
    data = []
    for dir_name in os.listdir(data_dir):
        # Skip directories that aren't in our valid classes
        if dir_name not in valid_classes:
            continue
            
        # Assign labels based on class names
        if dir_name == "Normal":
            label = 0
        elif dir_name == "Pneumonia-Bacterial":
            label = 1
        elif dir_name == "Pneumonia-Viral":
            label = 2
            
        # Add images to dataframe
        for fname in os.listdir(os.path.join(data_dir, dir_name)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append({
                    "image_path": os.path.join(data_dir, dir_name, fname), 
                    "label": label
                })
    
    return pd.DataFrame(data)

# Function to resize image arrays
def resize_image_array(image_path):
    try:
        # Open as grayscale and keep as single channel for efficiency
        img = Image.open(image_path).convert('L').resize((128, 128))
        return np.asarray(img)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        # Return a blank image if there's an error
        return np.zeros((128, 128), dtype=np.uint8)

# Function to plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Function to plot data distribution
def plot_data_distribution(class_counts, label_map):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(list(label_map.values()), class_counts)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution in Dataset')
    plt.tight_layout()
    plt.show()

# Function to plot sample images
def plot_sample_images(df, label_map, samples_per_class=3):
    plt.figure(figsize=(15, 5))
    for i, class_label in enumerate(sorted(df['label'].unique())):
        samples = df[df['label'] == class_label].sample(min(samples_per_class, len(df[df['label'] == class_label])))
        
        for j, (_, sample) in enumerate(samples.iterrows()):
            plt.subplot(len(df['label'].unique()), samples_per_class, i*samples_per_class + j + 1)
            plt.imshow(sample['image'], cmap='gray')
            plt.title(f"{label_map[class_label]}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Sample Images from Each Class", y=1.05)
    plt.show()

# Function to plot ROC curves
def plot_roc_curves(y_test, y_pred_proba, num_classes, label_map):
    plt.figure(figsize=(10, 8))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, 
                 label=f'ROC curve for {label_map[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.show()
    
    # Return the average AUC
    return np.mean(list(roc_auc.values()))

# Function to create and train a custom CNN model
def create_and_train_custom_model(X_train, y_train, X_validate, y_validate, input_shape, num_classes, epochs=20):
    # Create a custom CNN model without requiring pretrained weights
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.1),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.1),
        
        # Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=3,
            verbose=1,
            factor=0.5,
            min_lr=0.00001
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Print model summary
    model.summary()
    
    # Fit the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_validate, y_validate),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# Main execution
def main():
    # Define path to the dataset
    data_dir = '/kaggle/input/curated-chest-xray-image-dataset-for-covid19/Curated X-Ray Dataset'
    
    # Set the maximum number of images per class for balancing
    max_images_per_class = 1600
    
    # Define class mapping
    label_map = {
        0: "Normal",
        1: "Pneumonia-Bacterial",
        2: "Pneumonia-Viral"
    }
    
    num_classes = len(label_map)
    print("Label mapping:", label_map)
    
    # Create dataframe from the dataset
    print("Creating dataframe from images...")
    df = create_dataframe(data_dir)
    
    # Check if we have data
    if len(df) == 0:
        print("No valid images found in the dataset. Please check the directory structure.")
        return
    
    print(f"Total images found: {len(df)}")
    
    # Parallelize resizing process
    print("Resizing images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        df['image'] = list(executor.map(resize_image_array, df['image_path']))
    
    # Count the number of images in each class
    class_counts = df['label'].value_counts().sort_index()
    
    # Print the dataset summary
    print("\nDataset Summary")
    print("-" * 60)
    print(f"{'Class Label':<15} {'Class Name':<30} {'Count':<10}")
    print("-" * 60)
    for label, name in label_map.items():
        count = class_counts.get(label, 0)
        print(f"{label:<15} {name:<30} {count:<10}")
    print("-" * 60)
    print(f"{'Total':<45} {len(df):<10}")
    
    # Visualize data distribution
    plot_data_distribution(class_counts, label_map)
    
    # Plot sample images
    plot_sample_images(df, label_map)
    
    # Use data augmentation to balance classes
    print("Balancing classes with data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Initialize augmented DataFrame
    augmented_df = pd.DataFrame(columns=['image_path', 'label', 'image'])
    
    # For each class, either sample down to max_images_per_class or augment up to it
    for class_label in range(num_classes):
        # Get images for this class
        class_df = df[df['label'] == class_label]
        class_count = len(class_df)
        
        if class_count == 0:
            print(f"Warning: No images found for class {label_map[class_label]}")
            continue
            
        print(f"Processing class {label_map[class_label]}: {class_count} images")
        
        if class_count > max_images_per_class:
            # If we have too many, sample down
            class_df = class_df.sample(max_images_per_class, random_state=42)
            augmented_df = pd.concat([augmented_df, class_df], ignore_index=True)
        else:
            # Add all existing images
            augmented_df = pd.concat([augmented_df, class_df], ignore_index=True)
            
            # If we need more, augment
            images_needed = max_images_per_class - class_count
            if images_needed > 0:
                print(f"Augmenting {images_needed} more images for {label_map[class_label]}")
                
                # Select images to augment (with replacement if needed)
                augment_source = class_df.sample(images_needed, replace=True)
                
                for _, row in augment_source.iterrows():
                    img_array = row['image']
                    # Need to convert grayscale to 3D array for ImageDataGenerator
                    image_tensor = np.expand_dims(img_array, axis=0)
                    image_tensor = np.expand_dims(image_tensor, axis=-1)
                    
                    # Generate an augmented image
                    aug_img = next(datagen.flow(image_tensor, batch_size=1))[0]
                    # Convert back to 2D array
                    aug_img = aug_img.squeeze().astype('uint8')
                    
                    # Add to dataframe
                    new_row = pd.DataFrame([{
                        'image_path': None, 
                        'label': class_label, 
                        'image': aug_img
                    }])
                    
                    augmented_df = pd.concat([augmented_df, new_row], ignore_index=True)
    
    # Update df to use the balanced dataset
    df = augmented_df
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Count the balanced classes
    balanced_counts = df['label'].value_counts().sort_index()
    print("\nBalanced Dataset Summary")
    print("-" * 60)
    for label, name in label_map.items():
        count = balanced_counts.get(label, 0)
        print(f"{label:<15} {name:<30} {count:<10}")
    print("-" * 60)
    print(f"{'Total':<45} {len(df):<10}")
    
    # Prepare data for training
    X = np.stack(df['image'].values)
    y = df['label'].values
    
    # Split into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42)
    
    # Normalize images and add channel dimension
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Add channel dimension for grayscale images
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    
    # Count the number of images per class in each split
    train_counts = np.sum(y_train, axis=0)
    val_counts = np.sum(y_val, axis=0)
    test_counts = np.sum(y_test, axis=0)
    
    # Print dataset split summary
    print("\nDataset Split Summary")
    print("-" * 90)
    print(f"{'Class Label':<15} {'Class Name':<30} {'Train':<10} {'Validation':<12} {'Test':<10} {'Total':<10}")
    print("-" * 90)
    for label, name in label_map.items():
        train_num = int(train_counts[label])
        val_num = int(val_counts[label])
        test_num = int(test_counts[label])
        total_num = train_num + val_num + test_num
        print(f"{label:<15} {name:<30} {train_num:<10} {val_num:<12} {test_num:<10} {total_num:<10}")
    print("-" * 90)
    total_images = len(y_train) + len(y_val) + len(y_test)
    print(f"{'Total':<46} {len(y_train):<10} {len(y_val):<12} {len(y_test):<10} {total_images:<10}")
    
    # Define input shape (grayscale, single channel)
    input_shape = (128, 128, 1)
    
    # Create and train the model
    print("\nTraining custom CNN model...")
    model, history = create_and_train_custom_model(
        X_train, y_train, X_val, y_val, input_shape, num_classes, epochs=30
    )
    
    # Visualize training history
    plot_training_history(history)
    
    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, list(label_map.values()))
    
    # Display classification report
    print("\nClassification Report:")
    class_report = classification_report(y_true, y_pred, target_names=list(label_map.values()))
    print(class_report)
    
    # Plot ROC curves
    avg_auc = plot_roc_curves(y_test, y_pred_proba, num_classes, label_map)
    print(f"Average AUC: {avg_auc:.4f}")
    
    # Try to save the model
    try:
        model.save('./x_ray_classifier_model.h5')
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Function to make predictions on new images
    def predict_image(image_path):
        # Load and preprocess the image
        img = resize_image_array(image_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.title("Test Image")
        plt.axis('off')
        plt.show()
        
        # Preprocess for prediction
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        
        # Predict
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = label_map[predicted_class]
        
        # Show prediction
        print(f"Predicted Class: {predicted_label}")
        print(f"Confidence: {prediction[0][predicted_class]:.4f}")
        
        # Show prediction distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(label_map.values()), y=prediction[0])
        plt.xlabel('Classes')
        plt.ylabel('Probability')
        plt.title('Prediction Probability Distribution')
        plt.tight_layout()
        plt.show()
    
    # Try to predict a sample image if available
    print("\nTrying to predict a sample image...")
    for class_name in label_map.values():
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            images = os.listdir(class_dir)
            if images:
                sample_path = os.path.join(class_dir, images[0])
                print(f"Testing with sample image: {sample_path}")
                predict_image(sample_path)
                break

if __name__ == "__main__":
    main()
