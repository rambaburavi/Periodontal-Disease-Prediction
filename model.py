import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

img_height, img_width = 128, 128
batch_size = 10
epochs = 10
k_folds = 5

aug_output_dir = r'C:\Users\Rambabu\OneDrive\Desktop\periodontal\augmented_output'
categories = ['penyakit-periodontal', 'penyakit-non-periodontal']
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.vgg16.preprocess_input(img_array)

image_paths, labels = [], []
for idx, category in enumerate(categories):
    folder = os.path.join(aug_output_dir, category)
    for img_name in os.listdir(folder):
        image_paths.append(os.path.join(folder, img_name))
        labels.append(idx)

print(f"\n Loaded {len(image_paths)} images for training.")

images = np.vstack([load_and_preprocess_image(p) for p in image_paths])
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
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

best_model = None
best_accuracy = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
    print(f"\n🔁 Training Fold {fold+1}/{k_folds}")
    model = create_model()
    history = model.fit(images[train_idx], labels[train_idx],
                        validation_data=(images[val_idx], labels[val_idx]),
                        epochs=epochs, batch_size=batch_size, verbose=1)

    val_loss, val_acc = model.evaluate(images[val_idx], labels[val_idx], verbose=0)
    print(f" Fold {fold+1} Accuracy: {val_acc:.4f}")

    if val_acc > best_accuracy:
        best_model = model
        best_accuracy = val_acc

model_save_path = r'C:\Users\Rambabu\OneDrive\Desktop\periodontal\periodontal_model.h5'
best_model.save(model_save_path)
print(f"\n Best model saved to: {model_save_path}")
