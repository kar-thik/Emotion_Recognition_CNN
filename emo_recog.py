from __future__ import print_function
import numpy as np 
import time
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image

class emotionRecog:
    def __init__(self):
        self.model = self.load_model()

    def load_model_cnn(self):
        json_file = open('./model_isseu.json', 'r')
        loaded_model_json = json_file.read() 
        json_file.close()
        model_cnn = model_from_json(loaded_model_json)
        model_cnn.load_weights('./model_isseu.h5')
        return model_cnn


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        exit()
    network = EmotionRecognition()
    if sys.argv[1] == 'start':
        import startCamera
    else:
        show_usage()
        exit()