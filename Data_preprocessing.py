import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_images_and_labels(image_dir, labels_path):
    images = []
    labels = []
    labels_df = pd.read_csv(labels_path)

    
    print("Labels DataFrame:")
    print(labels_df.head())

    for _, row in labels_df.iterrows():
        
        image_file = row['path'].strip()
        if image_file.startswith('train/'):
            image_file = image_file[len('train/'):]  
        
        
        image_path = os.path.normpath(os.path.join(image_dir, image_file))

        
        print(f"Loading image from: {image_path}")

        if os.path.isfile(image_path):  
            try:
                img = load_img(image_path, target_size=(224, 224))  
                img_array = img_to_array(img) / 255.0  
                images.append(img_array)
                labels.append(row['label'])  
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        else:
            print(f"Image not found: {image_path}")  

    return np.array(images), np.array(labels)


train_dir = 'data/train'  
labels_path = 'data/labels_path.csv'


X_train, y_train = load_images_and_labels(train_dir, labels_path)

print(f"Loaded {len(X_train)} images and {len(y_train)} labels.")