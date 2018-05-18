from __future__ import print_function
import numpy as np 
import time
import plaidml.keras 
plaidml.keras.install_backend()
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

with open("./data/fer2013/fer2013.csv") as f:
    content = f.readlines()
 
    lines = np.array(content)
 
num_of_instances = lines.size
print("number of instances: ",num_of_instances)

x_train, y_train, x_test, y_test = [], [], [], []
 
for i in range(1,num_of_instances):
 try:
  emotion, img, usage = lines[i].split(",")
   
  val = img.split(" ")
  pixels = np.array(val, 'float32')
 
  emotion = keras.utils.to_categorical(emotion, 7)
 
  if 'Training' in usage:
   y_train.append(emotion)
   x_train.append(pixels)
  elif 'PublicTest' in usage:
   y_test.append(emotion)
   x_test.append(pixels)
 except:
  print("", end="")

model = Sequential()
batch_size = 16
epochs = 3

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

x_train = np.array(x_train)
x_train = x_train.reshape(-1,48,48,1)

gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=16)
 
model.compile(loss='categorical_crossentropy'
, optimizer=keras.optimizers.Adam()
, metrics=['accuracy']
)

model.fit_generator(train_generator, steps_per_epoch=5, epochs=100)


train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])
 
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])