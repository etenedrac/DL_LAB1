from __future__ import division
import keras
import os
import tensorflow as tf
import cv2
import random
import gc
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from time import time
from keras.preprocessing.image import ImageDataGenerator

#x_train = np.load('x_train.npy')
#y_train = np.load('y_train.npy')
#x_valid = np.load('x_valid.npy')
#y_valid = np.load('y_valid.npy')
#x_test = np.load('x_test.npy')
#y_test = np.load('y_test.npy')

#Normalize data
#x_train = x_train.astype('float32')
#x_valid = x_valid.astype('float32')
#x_test = x_test.astype('float32')
#x_train = x_train / 255
#x_valid = x_valid / 255
#x_test = x_test / 255

from keras.utils import to_categorical 
#y_train = to_categorical(y_train, 10)
#y_valid = to_categorical(y_valid, 10)
#y_test = to_categorical(y_test, 10)

#MNIST resolution
img_rows, img_cols, channels = 32, 32, 3

#from keras import backend as K
#if K.image_data_format() == 'channels_first':
#    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
#    x_valid = x_valid.reshape(x_valid.shape[0], channels, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
#    input_shape = (channels, img_rows, img_cols)
#else:
#    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
#    x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, channels)
#    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
#    input_shape = (img_rows, img_cols, channels)

train_data = ImageDataGenerator()
valid_data = ImageDataGenerator()
test_data = ImageDataGenerator()

train_gen = train_data.flow_from_directory(
	'./train',
	target_size=(32,32),
	batch_size=128,
	class_mode='categorical')

valid_gen = valid_data.flow_from_directory(                                                                     './valid',                                                                                              target_size=(32,32),                                                                                  batch_size=128,                                                                                         class_mode='categorical') 

test_gen = test_data.flow_from_directory(
'./test',
target_size=(32,32),                                                                                  batch_size=128,                                                                                         class_mode='categorical')
  
#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout

nn = Sequential()
nn.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32,32,3)))
nn.add(Activation('relu'))
nn.add(Conv2D(32, (3, 3)))
nn.add(Activation('relu'))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Dropout(0.25))

nn.add(Conv2D(64, (3, 3), padding='same'))
nn.add(Activation('relu'))
nn.add(Conv2D(64, (3, 3)))
nn.add(Activation('relu'))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Dropout(0.25))

nn.add(Flatten())
nn.add(Dense(512))
nn.add(Activation('relu'))
nn.add(Dropout(0.5))
nn.add(Dense(10))
nn.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

nn.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

#Starting tensorboard
cbacks = []
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
cbacks.append(tensorboard)

modfile = './model%d.h5' % int(time())
mcheck = ModelCheckpoint(filepath=modfile, monitor='val_loss', verbose=0, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
cbacks.append(mcheck)

early = EarlyStopping(monitor='val_loss', patience=10)

cbacks.append(early)

#Start training
history = nn.fit_generator(train_gen,epochs=1000,steps_per_epoch=700, validation_data=valid_gen, validation_steps=700, verbose=0, callbacks=cbacks)

#Evaluate the model with test set
#score = nn.evaluate(x_test, y_test, verbose=0)
#print('test loss:', score[0])
#print('test accuracy:', score[1])

##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('5_cnn_accuracy.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('5_cnn_loss.pdf')

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
#Compute probabilities
Y_pred = nn.predict(x_test)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print( 'Analysis of results' )
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))
