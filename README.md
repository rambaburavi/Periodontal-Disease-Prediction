# Periodontal Disease Prediction
This project uses a Convolutional Neural Network (CNN) to predict periodontal (gum) disease from panoramic dental images. It applies **K-Fold Cross-Validation** to test model performance properly and performs **binary classification** (disease or no disease).

---

## Project Files
- `model.py` – Trains the CNN model using K-Fold cross-validation and saves the best model.
- `predict.py` – Loads the saved model and predicts the class (periodontal or not) for a new image.
- `app.py` – Flask-based web app that lets users upload a dental image and get a prediction.

---

## Features
- **Preprocessing**: All images are resized to 128x128 and normalized for VGG16 compatibility.
- **CNN Architecture**: Built with convolutional, pooling, and dense layers.
- **K-Fold Cross-Validation**: Data is split into 5 folds for better model evaluation.
- **Evaluation Metrics**: Accuracy, Precision, and Recall are calculated for each fold.
- **Training Visualization**: Plots for accuracy and loss over epochs.

---

## Dataset Structure
dataset/

│
├── penyakit-periodontal/ # Images with periodontal disease
└── penyakit-non-periodontal/ # Images without the disease

- All images are RGB and resized to **128x128 pixels**.

---

## Model Details
- **Layers**: Conv2D → ReLU → MaxPooling → Dense → Dropout → Output
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Final Activation**: Sigmoid (for binary output)

---

## Installation & Usage

### 1. Install Required Libraries
pip install tensorflow numpy matplotlib scikit-learn pandas flask
### 2. Train the model
python model.py
### 3. Predict for a new image
python predict.py --image path_to_image.jpg
### 4. Run web app
python app.py

