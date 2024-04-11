import os
import numpy as np
from keras.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

indices_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'indices.txt')
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'mask_detector.h5')


face_model = load_model(model_path, compile=False)


with open(indices_path, 'r') as f:
    class_names = [a[:-1].split('-')[1] for a in f.readlines()]
    f.close()


def classify(image, model = face_model, classes = class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """

    img_gen = ImageDataGenerator(rescale=1/255)

    img = image.resize((300, 300))
    img_arr = np.array(img)
    img_arr = img_arr[np.newaxis, :]
    img_arr = img_arr.astype("float")
    img_arr = img_gen.standardize(img_arr)
    prob = model(img_arr)

    index = 1 if prob > 0.5 else 0
    classes = classes[index]
    
    return classes
