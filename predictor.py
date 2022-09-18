import tensorflow as tf
import numpy as np
import cv2

import os

class Predictor():
    model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            model_path = os.getenv('SAVED_MODEL_PATH', '/home/condados/workarea/personal-easy-clocking/model/model.savedmodel')
            cls.model = tf.saved_model.load(model_path)
        return cls.model
    
    @classmethod
    def predict(cls, image):
        model = cls.get_model()
        prep_image = cls.preprocessing(image)
        pred = model(prep_image)
        pos_pred = cls.posprocessing(pred)
        return pos_pred

    @classmethod
    def preprocessing(cls, image):
        prep_image = cv2.resize(image, [224, 224]).astype(np.float32)
        prep_image = (prep_image / 127) - 1.0 #normalize to [-1, 1]
        prep_image = np.expand_dims(prep_image, axis=0)
        return prep_image

    @classmethod
    def posprocessing(cls, model_output):
        np_pred = model_output.numpy()
        pos_pred = np.argmax(np_pred)
        return pos_pred