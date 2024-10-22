import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json  

def load_images_and_labels(image_dir, labels_path):
    images = []
    labels = []
    labels_df = pd.read_csv(labels_path)

    for _, row in labels_df.iterrows():
        
        image_path = os.path.normpath(os.path.join(image_dir, row['path'].replace('train/', '')))
        
        print(f"Loading image from: {image_path}")  
        if os.path.exists(image_path):  
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0  
            images.append(img_array)
            labels.append(row['label'])  
        else:
            print(f"Image not found: {image_path}")  

    return np.array(images), np.array(labels)

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    
    train_dir = 'data/train'  
    labels_path = 'data/labels_path.csv'
    
    
    X, y = load_images_and_labels(train_dir, labels_path)

    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = build_model(input_shape=(224, 224, 3))

    
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    
    
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy:.2f}")

    
    y_pred = (model.predict(X_val) > 0.5).astype(int).flatten()  

    
    predictions = {f"imagen_{i + 1}.jpeg": int(pred) for i, pred in enumerate(y_pred)}
    
    
    predictions_path = 'predictions/predictions.json'
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)  

    print("Saving predictions to JSON...")
    with open(predictions_path, 'w') as json_file:
        json.dump({"target": predictions}, json_file, indent=4)
    print("Predictions saved to predictions.json.")