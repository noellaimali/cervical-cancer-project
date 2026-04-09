import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def main():
    # 1. Create directory if it doesn't exist
    target_dir = 'invalid cells'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    # 2. Load CIFAR-10 data
    print("Loading CIFAR-10 data...")
    # Using local loading or downloading if needed
    (x_train, y_train), _ = cifar10.load_data()

    # 3. Save a selection of images
    # We'll save 200 images to have a decent amount for the 'invalid' class
    num_to_save = 200
    print(f"Saving {num_to_save} images to '{target_dir}'...")
    
    for i in range(num_to_save):
        img_path = os.path.join(target_dir, f'invalid_{i}.png')
        # CIFAR-10 is RGB, OpenCV wants BGR for imwrite
        img_bgr = cv2.cvtColor(x_train[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_bgr)
        
    print("Done! 'invalid cells' folder is now populated.")

if __name__ == "__main__":
    main()
