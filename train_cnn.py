
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import cv2
import matplotlib.pyplot as plt
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 30
BASE_DIR = os.getcwd()

def get_image_paths_and_labels():
    """
    Scans the directory for images. It looks for 'normal cells' and 'abnormal cells' folders.
    If a folder has a 'CROPPED' subfolder, it uses that instead of the main folder
    to avoid using full slide BMPs.
    Returns image paths and corresponding labels.
    """
    image_paths = []
    labels = []

    print("Scanning for class folders in the base directory...")
    
    # We explicitly define the classes we expect based on the project structure
    target_classes = ['normal cells', 'abnormal cells', 'invalid cells']
    
    for class_name in target_classes:
        folder_path = os.path.join(BASE_DIR, class_name)
        if not os.path.exists(folder_path):
            print(f" - Class '{class_name}': Folder NOT found at {folder_path}")
            continue

        # Look for a 'CROPPED' subfolder first, as it contains the individual cell images
        cropped_path = os.path.join(folder_path, 'CROPPED')
        search_path = cropped_path if os.path.exists(cropped_path) else folder_path
        
        print(f" - Scanning '{class_name}' in: {search_path}")
        
        extensions = ['*.bmp', '*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG']
        class_files = []
        for ext in extensions:
            # Use recursive glob or just glob depending on if we are in CROPPED
            files = glob.glob(os.path.join(search_path, ext))
            for f in files:
                # Basic check: skip files that are too large (likely slides, not single cells)
                # Medical slides are ~9MB, cropped cells are ~40KB. Threshold at 500KB.
                if os.path.getsize(f) < 500 * 1024:
                    class_files.append(f)
        
        if class_files:
            print(f" - Class '{class_name}': Found {len(class_files)} relevant images.")
            for img_path in class_files:
                image_paths.append(img_path)
                labels.append(class_name)
        else:
            print(f" - Class '{class_name}': No suitable images found.")
            
    return image_paths, labels

def load_and_preprocess_images(image_paths, labels):
    """
    Loads images from paths, resizes them, and normalizes pixel values.
    """
    data = []
    
    print("\nLoading and preprocessing images...")
    for i, img_path in enumerate(image_paths):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{len(image_paths)} images", end='\r')
            
        try:
            # Load image in color
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping corrupt image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            
            # Normalize to [0, 1]
            img = img.astype('float32') / 255.0
            
            data.append(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            
    print(f"Finished processing {len(data)} images.")
    return np.array(data)

def build_cnn_model(input_shape, num_classes):
    """
    Builds a CNN model with Data Augmentation for better accuracy.
    """
    model = models.Sequential()
    
    # Data Augmentation Layer
    model.add(layers.Input(shape=input_shape))
    model.add(layers.RandomFlip("horizontal_and_vertical"))
    model.add(layers.RandomRotation(0.2))
    model.add(layers.RandomZoom(0.2))
    
    # Convolutional Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Block 4 (New for 128x128)
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5)) # Regularization
    
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
        
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])
    
    return model

def main():
    # 1. Load Data
    image_paths, labels = get_image_paths_and_labels()
    
    if not image_paths:
        print("No images found! Please check your directory structure.")
        return

    # 2. Encode Labels
    # Map the folder names to the requested output labels
    label_map = {
        'abnormal cells': 'CANCEROUS',
        'normal cells': 'NON-CANCEROUS',
        'invalid cells': 'INVALID'
    }
    labels = [label_map.get(l, l) for l in labels]
    
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    unique_labels = le.classes_
    print(f"\nLabels mapped: {label_map}")
    print(f"Classes found: {unique_labels}")
    
    X = load_and_preprocess_images(image_paths, labels)
    y = labels_encoded
    
    num_classes = len(unique_labels)
    
    # Handle single class case
    if num_classes < 2:
        print("\nWARNING: Only 1 class found. Training a classifier requires at least 2 classes.")
        print("We will proceed with a dummy split just to demonstrate the code, but accuracy will be meaningless.")
        # We can't use 'stratify' with 1 class
        stratify_param = None
    else:
        stratify_param = y

    # 3. Model Preparation
    if num_classes > 2:
        y = to_categorical(y)

    if len(X) == 0:
        print("No valid images loaded. Exiting.")
        return

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_param)
    
    print(f"\nTraining Samples: {X_train.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")
    
    # Calculate class weights to handle imbalance
    # (Many abnormal cells vs few normal cells)
    if num_classes >= 2:
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels_encoded),
            y=labels_encoded
        )
        class_weights = dict(enumerate(weights))
        print(f"Calculated class weights: {class_weights}")
    else:
        class_weights = None

    # 5. Build and Train Model
    model = build_cnn_model((IMG_HEIGHT, IMG_WIDTH, 3), num_classes if num_classes > 2 else 2)
    model.summary()
    
    history = model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=(X_test, y_test),
        class_weight=class_weights
    )
    
    # 6. Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save classes
    with open('classes.json', 'w') as f:
        json.dump(unique_labels.tolist(), f)
    print("Classes saved to classes.json")

    # 7. Save Model
    model.save('cervical_cell_classifier.h5')
    print("Model saved to cervical_cell_classifier.h5")

    # 8. Plot and Save Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")

    # 9. Generate and Save Confusion Matrix
    y_pred = model.predict(X_test)
    if num_classes == 2:
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        y_true_classes = y_test.astype(int).flatten()
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    main()