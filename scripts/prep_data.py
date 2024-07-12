import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_images_from_folder(folder, img_size):
    images = []
    labels = []
    label_names = sorted(os.listdir(folder))  # Sorted
    
    for label in label_names:
        label_path = os.path.join(folder, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label_names.index(label))
    return np.array(images), np.array(labels), label_names

def prepare_data(raw_train_dir, processed_data_dir, img_size=224, test_size=0.2):
    print("Loading training images from folder...")
    X, y, label_names = load_images_from_folder(raw_train_dir, img_size)
    
    print(f"Splitting data into train and test sets with test size = {test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    
    print("Saving processed data...")
    np.save(os.path.join(processed_data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_data_dir, 'y_test.npy'), y_test)
    with open(os.path.join(processed_data_dir, 'label_names.txt'), 'w') as f:
        for label in label_names:
            f.write(f"{label}\n")
    
    print("Data preparation completed.")
    
    # Debugging information
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Unique labels in y_train: {np.unique(y_train)}")
    print(f"Unique labels in y_test: {np.unique(y_test)}")
    
    # Display a few sample images with their labels
    for i in range(5):
        plt.imshow(cv2.cvtColor(X_train[i], cv2.COLOR_BGR2RGB))
        plt.title(f"Label: {label_names[y_train[i]]}")
        plt.show()

if __name__ == "__main__":
    # Paths
    raw_train_dir = 'data\\raw\\asl_alphabet_train'
    processed_data_dir = 'data\\processed'
    
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    
    prepare_data(raw_train_dir, processed_data_dir)
