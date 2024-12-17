# Periodontal-Disease-Prediction
Periodontal Disease Prediction with TensorFlow

This project focuses on building a convolutional neural network (CNN) to predict periodontal diseases using panoramic dental images. The model employs K-Fold Cross-Validation for robust evaluation and achieves binary classification between periodontal and non-periodontal conditions.


---

Features

Preprocessing: Image resizing, normalization, and preprocessing for VGG16 compatibility.

Model Architecture: A CNN with multiple convolutional and pooling layers, followed by dense layers for classification.

K-Fold Cross-Validation: Splits the data into 5 folds to ensure unbiased evaluation.

Metrics: Reports Accuracy, Precision, and Recall for each fold.

Visualization: Plots training and validation accuracy and loss over epochs.



---


Dataset Structure:

penyakit-periodontal/ (Periodontal disease images)

penyakit-non-periodontal/ (Non-periodontal images)


Input Size: 128x128 pixels, RGB.



---

Model Details

Layers:

Convolutional layers with ReLU activation

Max pooling layers

Dense layers with Dropout for regularization


Optimizer: Adam

Loss Function: Binary Crossentropy

Activation Function: Sigmoid for binary classification.



---

Installation and Usage

Prerequisites

Python 3.8+

TensorFlow 2.x

NumPy

Matplotlib

scikit-learn

pandas

---

Results

K-Fold Cross-Validation

Average Accuracy: 0.6900

Average Precision: 0.6662

Average Recall: 0.7667


Training history plots for Accuracy and Loss over epochs are also generated.


---

Future Work

Augment the dataset with more samples.

Explore pre-trained models for transfer learning.

Deploy the model using Flask/Django for real-world applications.



---

License

This project is licensed under the MIT License.
