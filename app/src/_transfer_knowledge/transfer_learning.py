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

    return dataset_train, dataset_test, dataset_train.class_indices


def network_builder(activation, dropout_rate, dataset_train, fine_tuning, percent_to_finetuning, epochs):

    model_base = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=tf.keras.layers.Input(shape=(256, 256, 3))
    )

    n_layers_to_finetuning = int(
        percent_to_finetuning * len(model_base.layers))

    if fine_tuning == True:
        for layer in model_base.layers[:n_layers_to_finetuning]:
            layer.trainable = True

    else:
        for layer in model_base.layers:
            layer.trainable = False

    builder = tf.keras.layers
    model_head = model_base.output

    model_head = builder.GlobalAveragePooling2D()(model_head)

    n_neurons = int((model_base.output.shape[3] + 2)/2)

    model_head = builder.Dense(
        units=n_neurons,
        activation=activation)(model_head)

    model_head = builder.Dropout(rate=dropout_rate)(model_head)

    model_head = builder.Dense(
        units=n_neurons,
        activation=activation)(model_head)

    model_head = builder.Dropout(rate=dropout_rate)(model_head)

    model_head = builder.Dense(
        units=2,
        activation='softmax')(model_head)

    network = tf.keras.models.Model(
        inputs=model_base.input, outputs=model_head)

    network.compile(optimizer='Adam',
                    loss='categorical_crossentropy', metrics=['accuracy'])

    history = network.fit(dataset_train, epochs=epochs)

    model_json = network.to_json()
    with open('network.json', 'w') as f:
        f.write(model_json)

    save_model(network, 'weights.hdf5')

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


def predict_img(img, len_resize, network, class_index):
    img = cv2.imread(img)
    img = cv2.resize(img, (len_resize[1], len_resize[2]))
    img = img/255
    img = img.reshape(len_resize[0], len_resize[1],
                      len_resize[2], len_resize[3])

    prediction = network.predict(img)
    prediction = np.argmax(prediction)

    class_index = {value: key for key, value in class_index.items()}

    return prediction, class_index[prediction]


def runner(img_path_to_detect, dataset_train_path, dataset_test_path, train=False):
    try:

        dataset_train, dataset_test, class_index = preprocessing(
            dataset_train_path,
            dataset_test_path,
            batch_size=8,
            target_size=(64, 64),
            class_mode='categorical',
            shuffle=True
        )

        if train == True:
            network_builder(
                activation='relu',
                dropout_rate=0.2,
                dataset_train=dataset_train,
                fine_tuning=True,
                percent_to_finetuning=20,
                epochs=50
            )

        model_loaded = load_model()

        metrics = get_metrics(model_loaded, dataset_test=dataset_test)

        output = predict_img(
            img_path_to_detect,
            [-1, 64, 64, 3],
            model_loaded,
            class_index
        )
    except:
        return 'error'

    return metrics, output


""" output_network = runner(
    img_path_to_detect="../../data/Datasets/cat_dog_2/cat_dog_2/test_set/cat/cat.3500.jpg",
    dataset_train_path="../../data/Datasets/cat_dog_2/cat_dog_2/training_set",
    dataset_test_path="../../data/Datasets/cat_dog_2/cat_dog_2/test_set",
    train=True
) """

metrics, predictions = runner(
    img_path_to_detect="../../data/Datasets/cat_dog_2/cat_dog_2/test_set/bart/bart3.bmp",
    dataset_train_path="../../data/Datasets/cat_dog_2/cat_dog_2/training_set",
    dataset_test_path="../../data/Datasets/cat_dog_2/cat_dog_2/test_set",
    train=True
)
