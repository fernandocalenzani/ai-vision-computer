import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import save_model

def preprocessing():
    _paths = []
    imgs__ = []
    class__ = []
    _path = "../../data/Datasets/homer_bart_1/homer_bart_1"
    _files = [os.path.join(_path, f) for f in sorted(os.listdir(_path))]
    _l, _h = 128, 128
    _scaler = MinMaxScaler()

    for _paths in _files:
        try:
            _image = cv2.imread(_paths)
            (H, W) = _image.shape[:2]
        except:
            continue

        _image = cv2.resize(_image, (_l, _h))
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        _image = _image.ravel()

        imgs__.append(_image)

        _img_name = os.path.basename(os.path.normpath(_paths))

        if (_img_name.startswith('b')):
            _img_class = 0
        else:
            _img_class = 1

        class__.append(_img_class)

    x = np.asarray(imgs__)
    x = _scaler.fit_transform(x)
    y = np.asarray(class__)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1)

    return x_train, x_test, y_train, y_test, x, y


def model(input_shape, units):
    nn = tf.keras.models.Sequential()

    nn.add(tf.keras.layers.Dense(input_shape=input_shape,
                                 units=units, activation='relu'))

    nn.add(tf.keras.layers.Dense(input_shape=input_shape,
                                 units=units, activation='relu'))

    nn.add(tf.keras.layers.Dense(
        units=1, activation='sigmoid'))

    nn.compile(optimizer='Adam', loss='binary_crossentropy',
               metrics=['accuracy'])

    nn.summary()

    return nn


x_train, x_test, y_train, y_test, x, y = preprocessing()
model = model((16384,), 8193)
nn = model.fit(x_train, y_train, epochs=50)
nn_json = nn.to_json

with open('nn.json', 'w') as f:
    f.write(nn_json)

nn_saved = save_model(nn, 'weights.hdf5', nn_json)
