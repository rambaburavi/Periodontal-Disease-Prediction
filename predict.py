import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2


model_path = r'C:\Users\Rambabu\OneDrive\Desktop\periodontal\periodontal_model.h5'
img_path = r'C:\Users\Rambabu\OneDrive\Desktop\periodontal\xrayperi.png'
img_height, img_width = 128, 128

def is_valid_xray(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    avg_pixel = np.mean(img)
    std_dev = np.std(img)

    if avg_pixel > 180 or avg_pixel < 20:
        return False
    if std_dev < 10:
        return False

    h, w = img.shape
    if h < 100 or w < 100:
        return False

    return True

if not is_valid_xray(img_path):
    print(" This image is not a valid dental X-ray.")
else:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")

    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        print(" Prediction: Periodontal Disease")
    else:
        print(" Prediction: Non Periodontal Disease")
