import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import save_model
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


def preprocessing(path_train, path_test, batch_size, target_size, class_mode, shuffle):
    load = tf.keras.preprocessing.image

    gen_train = load.ImageDataGenerator(
        rescale=1./255, rotation_range=7, horizontal_flip=True, zoom_range=0.2)

    dataset_train = gen_train.flow_from_directory(
        path_train,
        target_size=target_size,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle
    )

    gen_test = load.ImageDataGenerator(rescale=1./255)

    dataset_test = gen_test.flow_from_directory(
        path_test,
        target_size=target_size,
        class_mode=class_mode,
        batch_size=1,
        shuffle=False
    )

    return dataset_train, dataset_test


def model(n_filters, kernel_size, activation, input_shape, pool_size, units, output_units, activation_output, dataset_train, epochs):
    builder = tf.keras.layers
    network = tf.keras.models.Sequential()

    network.add(builder.Conv2D(
        filters=n_filters,
        kernel_size=kernel_size,
        activation=activation,
        input_shape=input_shape
    ))
    network.add(builder.MaxPool2D(pool_size=pool_size))

    network.add(builder.Conv2D(
        filters=n_filters,
        kernel_size=kernel_size,
        activation=activation,
    ))
    network.add(builder.MaxPool2D(pool_size=pool_size))

    network.add(builder.Conv2D(
        filters=n_filters,
        kernel_size=kernel_size,
        activation=activation,
    ))

    network.add(builder.MaxPool2D(pool_size=pool_size))

    network.add(builder.Flatten())

    network.add(builder.Dense(units=units, activation=activation))

    network.add(builder.Dense(units=units, activation=activation))

    network.add(builder.Dense(units=output_units,
                activation=activation_output))

    network.compile(optimizer='Adam',
                    loss='categorical_crossentropy', metrics=['accuracy'])

    history = network.fit(dataset_train, epochs=epochs)

    model_json = network.to_json()
    with open('network.json', 'w') as f:
        f.write(model_json)

    nn_saved = save_model(network, 'weights.hdf5')

    return history


def get_metrics(model, dataset_test):
    try:
        metrics = {}

        predictions = model.predict(dataset_test)

        metrics['predictions'] = np.argmax(predictions, axis=1)
        metrics['accuracy_score'] = accuracy_score(
            dataset_test.classes, metrics['predictions'])

        metrics['cm'] = confusion_matrix(
            dataset_test.classes, metrics['predictions'])
        metrics['classification_report'] = classification_report(
            dataset_test.classes, metrics['predictions'])

        return metrics

    except:
        return metrics


def load_model():
    with open('network.json') as f:
        json_saved_model = f.read()

    network_loaded = tf.keras.models.model_from_json(json_saved_model)
    network_loaded.load_weights('weights.hdf5')

    network_loaded.compile(optimizer='Adam',
                           loss='categorical_crossentropy', metrics=['accuracy'])

    return network_loaded


def predict_img(img, len_resize, network):
    img = cv2.imread(img)
    img = cv2.resize(img, (len_resize[1], len_resize[2]))
    img = img/255
    img = img.reshape(len_resize[0], len_resize[1],
                      len_resize[2], len_resize[3])

    prediction = network.predict(img)
    prediction = np.argmax(prediction)

    return prediction


def runner(img_path_to_detect, dataset_train_path, dataset_test_path, train=False):
    dataset_train, dataset_test = preprocessing(
        dataset_train_path,
        dataset_test_path,
        batch_size=8,
        target_size=(64, 64),
        class_mode='categorical',
        shuffle=True
    )

    if train == True:
        model(
            n_filters=32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(64, 64, 3),
            pool_size=(2, 2),
            units=3137,
            output_units=2,
            activation_output='softmax',
            dataset_train=dataset_train,
            epochs=10
        )

    model_loaded = load_model()

    metrics = get_metrics(model_loaded, dataset_test=dataset_test)

    output = predict_img(
        img_path_to_detect,
        [-1, 64, 64, 3],
        model_loaded
    )

    return metrics, output


output_network = runner(
    img_path_to_detect="../../data/Datasets/cat_dog_2/cat_dog_2/test_set/cat/cat.3500.jpg",
    dataset_train_path="../../data/Datasets/cat_dog_2/cat_dog_2/training_set",
    dataset_test_path="../../data/Datasets/cat_dog_2/cat_dog_2/test_set",
    train=True
)
