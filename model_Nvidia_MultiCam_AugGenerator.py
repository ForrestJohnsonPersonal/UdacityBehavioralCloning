import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import sklearn
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import csv
import cv2
import math

from keras.models import load_model
import h5py
from keras import __version__ as keras_version
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

images = []
measurements = []
imagesCenter = []
measurementsCenter = []
imagesLeft = []
measurementsLeft = []
imagesRight = []
measurementsRight = []

def process_image(img):
 #   image_H, image_W, image_CH = 160, 320, 3  #### Per Nvidia model
 #   image = np.array(img[80:140,:]) #img[60:135, :, :]
 #   image = cv2.resize(image, (image_W, image_H), cv2.INTER_AREA)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (5,5), 0)
    return image

def Get3CarCameraImages(path, steering_center, HardBias=False, steering_HardBias=0):
    global images
    global measurements
    global correction
    global imagesCenter, measurementsCenter, imagesLeft, measurementsLeft, imagesRight, measurementsRight

    # create adjusted steering measurements for the side camera images
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    if (HardBias == True):
        steering_center = steering_center + steering_HardBias
        steering_left = steering_center - steering_HardBias/1.5
        steering_right = steering_right + steering_HardBias/1.5
    imagesCenter.append(np.asarray(process_image(cv2.imread(row[0])))) #Image.open(row[0]))))
    measurementsCenter.append(steering_center)
    imagesLeft.append(np.asarray(process_image(cv2.imread(row[1])))) #Image.open(row[1]))))
    measurementsLeft.append(steering_left)
    imagesRight.append(np.asarray(process_image(cv2.imread(row[2])))) #Image.open(row[2]))))
    measurementsRight.append(steering_right)
    # add images and angles to data set
#    images.extend(img_center, img_left, img_right)
#    measurements.extend(steering_center, steering_left, steering_right)
    return

correction = 0.4 # this is a parameter to tune

#Grab the main loop of driving data.      
with open('data1/dataFullLoop/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skips the first line
    for row in reader:
        # read in images from center, left and right cameras
        Get3CarCameraImages('data1/dataFullLoop/IMG/', float(row[3]))
#Grab the reverse main loop data.        
#with open('data1/dataFullLoopReverse/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    next(reader) #skips the first line
#    for row in reader:
        # read in images from center, left and right cameras
 #       Get3CarCameraImages('data1/dataFullLoopReverse/IMG/', float(row[3]))

#Grab the optional main loop of driving data.      
#with open('data1/dataFullOptionalLoop/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    next(reader) #skips the first line
#    for row in reader:
        # read in images from center, left and right cameras
#        Get3CarCameraImages('data1/dataFullOptionalLoop/IMG/', float(row[3]))
#Grab the Bridge data.        
with open('data1/dataBridge1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skips the first line
    for row in reader:
        # read in images from center, left and right cameras
        Get3CarCameraImages('data1/dataBridge1/IMG/', float(row[3]))     
#Grab the AvoidBridgeSide data.        
with open('data1/dataAvoidBridgeSideL/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skips the first line
    for row in reader:
        # read in images from center, left and right cameras
        Get3CarCameraImages('data1/dataAvoidBridgeSideL/IMG/', float(row[3]), True, 0.6)# increase as we are doing the harder correction data below.
#with open('data1/dataAvoidBridgeSide/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skips the first line
    for row in reader:
        # read in images from center, left and right cameras
        Get3CarCameraImages('data1/dataAvoidBridgeSide/IMG/', float(row[3]), True, -0.6)# increase as we are doing the harder correction data below.          
#Grab the AvoidOffRoad data.        
with open('data1/dataAvoidOffLeft/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skips the first line
    for row in reader:
        # read in images from center, left and right cameras
        Get3CarCameraImages('data1/dataAvoidOffLeft/IMG/', float(row[3]), True, 0.6)# increase as we are doing the harder correction data below.
with open('data1/dataAvoidOffRight/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skips the first line
    for row in reader:
        # read in images from center, left and right cameras
        Get3CarCameraImages('data1/dataAvoidOffRight/IMG/', float(row[3]), True, -0.6)# increase as we are doing the harder correction data below.        

X_train = np.concatenate((np.array(imagesCenter), np.array(imagesLeft), np.array(imagesRight)), axis=0)
y_train = np.concatenate((np.array(measurementsCenter), np.array(measurementsLeft), np.array(measurementsRight)), axis=0)
#Get normal plus left/right flipped data.
X_train = np.concatenate((X_train, np.fliplr(X_train)), axis=0)
y_train = np.concatenate((y_train, np.negative(y_train)), axis=0)
print (X_train.shape)

batch_size=64
validation_split=0.2

X_train_samples, X_validation_samples, y_train_samples, y_validation_samples = train_test_split(X_train, y_train, test_size=validation_split)
print (X_train_samples.shape)
print (X_validation_samples.shape)
trainingsize = len(X_train_samples)
validate_size = len(X_validation_samples)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D, Dropout, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers
from keras.optimizers import Adam

Adam(lr = 0.001)#changing from default of 0.01

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(filters=24,kernel_size=5,strides=2,activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(Conv2D(filters=36,kernel_size=5,strides=2,activation="relu", kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(Conv2D(filters=48,kernel_size=5,strides=2,activation="relu", kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(Convolution2D(64,3,3, activation="relu", kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(Convolution2D(64,3,3, activation="relu", kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.5))
model.add(Dense(50, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.5))
model.add(Dense(10, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(Dense(1))

def generator(samplesImage, samplesAngle, batch_size=32):
    num_samples = len(samplesImage)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samplesImage, samplesAngle)
        for offset in range(0, num_samples, batch_size):
            batch_samplesImage = samplesImage[offset:offset+batch_size]
            batch_samplesAngle = samplesAngle[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sampleImage in batch_samplesImage:
                images.append(batch_sampleImage)
            for batch_sampleAngle in batch_samplesAngle:
                angles.append(batch_sampleAngle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(X_train_samples, y_train_samples, batch_size=batch_size)
validation_generator = generator(X_validation_samples, y_validation_samples, batch_size=batch_size)

model.compile(loss='mse', optimizer='adam')#, steps_per_execution = batchSize)
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(X_train_samples)/batch_size), 
            validation_data=validation_generator, validation_steps=math.ceil(len(X_validation_samples)/batch_size), epochs=4, verbose=1)
#history_object = model.fit(X_train, y_train, validation_split=validation_split, shuffle=True, epochs=10,
#          steps_per_epoch=steps_per_epoch,
#          validation_steps=validation_steps)

model.save('model_Nvidia_MultiCam_AugGenerator5.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('lossimage.png')


    