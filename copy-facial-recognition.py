from cv2 import VideoCapture, resize, INTER_AREA, CAP_PROP_EXPOSURE, CAP_PROP_AUTO_EXPOSURE, cvtColor, CascadeClassifier, rectangle, imshow, imwrite, waitKey, COLOR_BGR2GRAY
from time import time, sleep
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import csv
import pandas as pd
start = time()

# Model Architecture
model = Sequential()
model.add(Conv2D(input_shape=(48, 48, 3), filters=64,
          kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
          padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Flatten())
model.add(Dense(units=500, activation="relu"))
model.add(Dropout(0.30))
model.add(Dense(units=500, activation="relu"))
model.add(Dropout(0.25))
# model.add(Dense(units=500,activation="relu"))
model.add(Dense(units=6, activation="softmax"))
# model.summary()

model.load_weights('./sentiment_model.h5')
# model.save_weights('./sentiment_model.h5')


vid = VideoCapture(0)
vid.set(CAP_PROP_AUTO_EXPOSURE, 0)
# vid.set(CAP_PROP_EXPOSURE, 40)

# with open('goal.csv', 'w', newline=) as file:
#     writer = csv.writer(file)
#     writer.writerow(["SN","Date","Mood"])

# creating a data frame
df = pd.DataFrame([['28/02/2023', '21:23:41', 2]],
                  columns=['Date', 'Time', 'Mood'])

# writing data frame to a CSV file
df.to_csv('person.csv', index=False)
print(pd.read_csv("person.csv"))


count = 0
while (count != 5):
    sTime = time()
    # print(sTime)
    t = datetime.now()
    print(t)
    current_time = t.strftime("%H:%M:%S")
    current_date = t.strftime("%d/%m/%Y")
    print("Current Date and Time = ", current_time)

    sleep(1)
    ret, frame = vid.read()
    img = frame

    # Convert into grayscale
    # gray = cvtColor(img, COLOR_BGR2GRAY)

    # Load the cascade
    face_cascade = CascadeClassifier('./haarface.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(img, 1.1, 1)

    dim = (48, 48)
    # faces = face_cascade.detectMultiScale
    # print(faces)
    x, y, w, h = faces[0]

    sphoto = img[y:y + h, x:x + w]
    resized = resize(sphoto, dim, interpolation=INTER_AREA)
    imwrite("face_0.jpg", sphoto)
    resized = resized.astype('float32') / 255
    img2 = np.array(resized)
    img2 = img2.reshape(1, 48, 48, 3)
    print(img2.shape)
    # img2

    # prediction of the image sentiment
    prediction = model.predict(img2)

    Mood = prediction.argmax()
    print(prediction.argmax())

    df = pd.DataFrame({'Date': [current_date],
                       'Time': [current_time],
                       'Mood': [Mood]})
    df.to_csv('person.csv', mode='a', index=False, header=False)

    # Draw rectangle around the faces and crop the faces
    # for (x, y, w, h) in faces:
    #     # rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #     sphoto = img[y:y + h, x:x + w]
    #     # imshow("face", sphoto)
    #     resized = resize(sphoto, dim, interpolation=INTER_AREA)
    #     imwrite('./faces/face{}.jpg'.format(count), sphoto)

    # for (x, y, w, h) in faces:
    #     # rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #     faces = img[y:y + h, x:x + w]
    #     imshow("face", faces)
    #     imwrite("./faces/faces_{}".format(str(count)), faces)

    eTime = time()
    print("Cycle {} Time: ".format(count), round(eTime-sTime, 2))
    count += 1


# Display the output
# imwrite('emotion.jpg', img)
# imshow('img', img)
# cv2.imshow('resized', resized)
end = time()
print("Finished Time ", round(end - start, 2), "seconds")
waitKey()
