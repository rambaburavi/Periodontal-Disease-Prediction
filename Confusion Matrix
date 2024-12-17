import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import os
import pandas as pd
import zipfile
import matplotlib.pyplot as plt

img_height, img_width = 128, 128
batch_size = 10
epochs = 10
k_folds = 5  

zip_path = '/archive.zip'  
data_dir = '/content/data/images/dental-panoramic' 

if not os.path.exists(data_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/content/data/images')  

print("Directory structure after extraction:")
for root, dirs, files in os.walk(data_dir):
    print(f"Root: {root}")
    print(f"Dirs: {dirs}")
    print(f"Files: {files}")
    print("-" * 30)

categories = ['penyakit-periodontal', 'penyakit-non-periodontal']

for category in categories:
    class_dir = os.path.join(data_dir, category)
    if not os.path.isdir(class_dir):
        raise ValueError(f"Directory {class_dir} not found. Verify data_dir and category names.")

image_paths = []
labels = []
for i, category in enumerate(categories):
    class_dir = os.path.join(data_dir, category)
    print(f"Loading images from {class_dir}...")  

    for img_name in os.listdir(class_dir):
        image_paths.append(os.path.join(class_dir, img_name))
        labels.append(i)  # 0 for periodontal, 1 for non-periodontal

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import KFold

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    return preprocess_input(img_array)  

print("Loading and preprocessing images...")
images = np.vstack([load_and_preprocess_image(path) for path in image_paths])
labels = np.array(labels)

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout for regularization
        Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

fold_no = 1
all_accuracy = []
all_precision = []
all_recall = []

for train_idx, val_idx in kf.split(images):
    print(f"Training fold {fold_no}/{k_folds}")

    train_images, val_images = images[train_idx], images[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    model = create_model()

    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
    print(f"Validation accuracy for fold {fold_no}: {val_accuracy:.4f}")

    all_accuracy.append(val_accuracy)

    val_predictions = (model.predict(val_images) > 0.5).astype('int32')
    report = classification_report(val_labels, val_predictions, output_dict=True)
    all_precision.append(report['1']['precision'])
    all_recall.append(report['1']['recall'])

    fold_no += 1

avg_accuracy = np.mean(all_accuracy)
avg_precision = np.mean(all_precision)
avg_recall = np.mean(all_recall)

print("\nK-Fold Cross-Validation Results:")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
