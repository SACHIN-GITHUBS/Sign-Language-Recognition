SLR.ipynb


from google.colab import drive
drive.mount('/content/drive/')

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import itertools
import random
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
warnings.simplefilter(action='ignore', category=FutureWarning)

train_path = r'/content/drive/MyDrive/train'
test_path = r'/content/drive/MyDrive/test1'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, 
               target_size=(64,64),
                class_mode='categorical', 
                batch_size=120,shuffle=True)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path,
               target_size=(64,64),
                class_mode='categorical',
               batch_size=30, shuffle=True)

imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 26, figsize=(30,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plotImages(imgs)
print(imgs.shape)
print(labels)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

#model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(128,activation ="relu"))
#model.add(Dropout(0.1))
model.add(Dense(26,activation ="softmax"))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

history2 = model.fit(train_batches, 
                     epochs=8, 
                     callbacks=[reduce_lr, early_stop],
                     validation_data = test_batches)

model.save('sachin')

scores = model.evaluate(imgs, labels, verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

# Commented out IPython magic to ensure Python compatibility.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
# %matplotlib inline
import tensorflow as tf
import matplotlib.image as img
import collections
from collections import defaultdict
from shutil import copy
from shutil import copytree, rmtree
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import models
import cv2
from keras.applications.imagenet_utils import decode_predictions
import glob
from PIL import Image

print(imgs.shape)
history2.history

modell = load_model('sachin')

imgs, labels = next(test_batches)

char_array = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

predictions = modell.predict(imgs, verbose=0)
print("predictions on a small set of test data--")
print("")
for ind, i in enumerate(predictions):
    plt.imshow(imgs[ind])
    plt.axis('off')
    print(char_array[np.argmax(i)],"(" +str(np.argmax(i))+ ")")   
    plt.show()

def predict_class(images, show = True):
  for img in images:
    img = image.load_img(img, target_size=(64,64))
    img =  image.img_to_array(img)             
    img = np.expand_dims(img, axis=0)         
    #img = np.array([img])
    img = tf.keras.applications.vgg16.preprocess_input(img)

    pred = modell.predict(img)
    index = np.argmax(pred)
    pred_value = char_array[index]
    print(pred)
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        print(pred_value, "("+str(index)+")")
        plt.show()

images = []
##images.append('/content/drive/MyDrive/test/A/A2511.jpg')

##images.append('/content/drive/MyDrive/test/B/B2568.jpg')

##images.append('/content/drive/MyDrive/test/C/C2528.jpg')

##images.append('/content/drive/MyDrive/test/I/I2528.jpg')

##images.append('/content/drive/MyDrive/test/O/O2530.jpg')

##images.append('/content/drive/MyDrive/test/J/J2713.jpg')

##images.append('/content/drive/MyDrive/test/K/K2713.jpg')
images.append('/content/drive/MyDrive/test1/W/W1053.jpg')
images.append('/content/drive/MyDrive/test1/V/V1010.jpg')
images.append('/content/drive/MyDrive/test1/O/O1010.jpg')
predict_class(images, True)


