from __future__ import print_function
import numpy as np 
import time
import np_utils
import plaidml.keras 
plaidml.keras.install_backend()
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers as opt
from keras import utils as ut

model = Sequential()

from keras.layers import Dense

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


with open("./data/fer2013/fer2013.csv") as f:
    content = f.readlines()
    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)


x_train, y_train, x_test, y_test = [], [], [], []
for i in range(1,35888):
    try:
        emotion, img, usage = lines[i].split(",")

        val = img.split(" ")
        pixels = np.array(val, 'float32')
        emotion = ut.to_categorical(emotion, 7)      
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
        print("", end="")

print (type(emotion))
model = Sequential()
 
#1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
 
#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 
#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 
model.add(Flatten())
 
#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
 
model.add(Dense(7, activation='softmax'))

x_train=np.array(x_train)

x_train =x_train.reshape(-1,48,48,1)

gen = ImageDataGenerator(rescale=1. / 255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)

train_generator = gen.flow(x_train, y_train,batch_size=512)


model.compile(loss="sparse_categorical_crossentropy",
              metrics=['accuracy'], optimizer=opt.Adam())
 
model.fit_generator(train_generator ,steps_per_epoch=5, epochs=500)

