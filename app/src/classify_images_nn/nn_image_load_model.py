import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from keras.models import save_model
from matplotlib import scale
from nn_image import preprocessing
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_model(path_nn_config, path_nn_weights):
    with open(path_nn_config) as f:
        json_saved_model = f.read()

    nn_load = tf.keras.models.model_from_json(json_saved_model)
    nn_load.load_weights(path_nn_weights)

    nn_load.compile(optimizer='Adam', loss='binary_crossentropy',
                    metrics=['accuracy'])

    return nn_load


model = load_model('nn.json', 'weights.hdf5')

x_train, x_test, y_train, y_test, x, y = preprocessing()

predictions = model.predict(x_test)

plt.plot(model.history['loss'])
plt.plot(model.history['accuracy'])

accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
sn.heatmap(cm, annot=True)
classification_report(y_test, predictions)


img = x_test[0]
img = scale.inverse_transform(img.reshape(-1, 1)).remove(128, 128)
predict_img = model.predict(img)
