from __future__ import print_function
import numpy as np 
import time
import plaidml.keras
plaidml.keras.install_backend()
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
import cv2
import sys
from numpy.random import seed
seed(1)

CASC_PATH = './haarcascade_files/haarcascade_frontalface_default.xml'
SIZE_FACE = 48
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

cascade_classifier = cv2.CascadeClassifier(CASC_PATH) 

json_file = open('./model_isseu.json', 'r')
loaded_model_json = json_file.read() 
json_file.close()
model_cnn = model_from_json(loaded_model_json)
model_cnn.load_weights('./model_isseu.h5')
print("Model and Weights succesfully loaded..")

def brighten(data,b):
     datab = data * b
     return datab 

def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor = 1.01,
        minNeighbors =8)
    # None is we don't find an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    face[0] -= 80
    face[1] -= 80
    face[2] += 80
    face[3] += 80
    #for (x, y, w, h) in face:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #face = [int(x *1.1) for x in face]
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    #cv2.imwrite("image.jpg",image)
    # Resize image to network size
    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
        image = np.expand_dims(image, axis = 0)
        image = np.expand_dims(image, axis = 4)
    except Exception:
        print("[+] Problem during resize")
        return None
    return image

video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
#  feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

while True:
      # Capture frame-by-frame
  ret, frame = video_capture.read()
  #cv2.imwrite("image1.jpg",format_image(frame))

  # Predict result with network
  result = model_cnn.predict(format_image(frame), batch_size=None)
  time.sleep(0.1)
  #print(EMOTIONS[np.argmax(result)])  
  # Draw face in frame


  # Write results in frame
  if result is not None:
    for index, emotion in enumerate(EMOTIONS):
      cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
      cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255,255,255), -1)
      #face_image = feelings_faces[result[0].tolist().index(max(result[0]))].tolist()
    # Ugly transparent fix
#    for c in range(0, 3):
#      frame[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :, 3] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)


  # Display the resulting frame
  cv2.imshow('Video', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
