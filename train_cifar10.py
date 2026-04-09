import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. Load Data
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 2. Preprocess Data
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"Training data shape: {x_train.shape}")
    print(f"Testing data shape: {x_test.shape}")

    # 3. Build the CNN Model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10) # 10 classes
    ])

    # 4. Compile the Model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 5. Train the Model
    print("\nStarting training...")
    history = model.fit(x_train, y_train, epochs=10, 
                        validation_data=(x_test, y_test))

    # 6. Evaluate the Model
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')

    # 7. Save the Model
    model.save('cifar10_model.h5')
    print("Model saved as cifar10_model.h5")

    # Optional: Plot training results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_history.png')
    print("Training history plot saved as training_history.png")

if __name__ == "__main__":
    main()
