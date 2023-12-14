import cv2
import numpy as np
import seaborn as sns
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


def network_builder(
    dataset_train,
    n_classes,
    epochs,
    n_filters,
    kernel_size,
    activation,
    padding,
    input_shape,
    pool_size,
    dropout_rate,
):
    builder = tf.keras.layers
    network = tf.keras.models.Sequential()

    """LAYER CONVOLUTIONAL 1------------------------------------"""
    network.add(builder.Conv2D(
        filters=n_filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        input_shape=input_shape
    ))
    network.add(builder.BatchNormalization())
    network.add(builder.Conv2D(
        filters=n_filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    ))
    network.add(builder.BatchNormalization())
    network.add(builder.MaxPool2D(pool_size=pool_size))
    network.add(builder.Dropout(dropout_rate))

    """LAYER CONVOLUTIONAL 2------------------------------------"""
    network.add(builder.Conv2D(
        filters=2*n_filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    ))
    network.add(builder.BatchNormalization())
    network.add(builder.Conv2D(
        filters=2*n_filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    ))
    network.add(builder.BatchNormalization())
    network.add(builder.MaxPool2D(pool_size=pool_size))
    network.add(builder.Dropout(dropout_rate))

    """LAYER CONVOLUTIONAL 3------------------------------------"""
    network.add(builder.Conv2D(
        filters=2*2*n_filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    ))
    network.add(builder.BatchNormalization())
    network.add(builder.Conv2D(
        filters=2*2*n_filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    ))
    network.add(builder.BatchNormalization())
    network.add(builder.MaxPool2D(pool_size=pool_size))
    network.add(builder.Dropout(dropout_rate))

    """LAYER CONVOLUTIONAL 4------------------------------------"""
    network.add(builder.Conv2D(
        filters=2*2*2*n_filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    ))
    network.add(builder.BatchNormalization())
    network.add(builder.Conv2D(
        filters=2*2*2*n_filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    ))
    network.add(builder.BatchNormalization())
    network.add(builder.MaxPool2D(pool_size=pool_size))
    network.add(builder.Dropout(dropout_rate))

    """LAYER CONVOLUTIONAL 5------------------------------------"""
    network.add(builder.Flatten())

    """LAYER DENSE 1------------------------------------"""
    network.add(builder.Dense(units=2*n_filters, activation=activation))
    network.add(builder.BatchNormalization())
    network.add(builder.Dropout(dropout_rate))

    """LAYER DENSE 2------------------------------------"""
    network.add(builder.Dense(units=2*n_filters, activation=activation))
    network.add(builder.BatchNormalization())
    network.add(builder.Dropout(dropout_rate))

    """LAYER DENSE OUTPUT------------------------------------"""
    network.add(builder.Dense(units=n_classes, activation='softmax'))

    """COMPILER------------------------------------"""
    network.compile(optimizer='Adam',
                    loss='categorical_crossentropy', metrics=['accuracy'])

    """TRAINING------------------------------------"""
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
            target_size=(48, 48),
            class_mode='categorical',
            shuffle=True
        )

        if train == True:
            network_builder(
                dataset_train=dataset_train,
                n_classes=len(class_index),
                epochs=50,
                n_filters=32,
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                input_shape=(48, 48, 3),
                pool_size=(2,2),
                dropout_rate=0.2
            )

        model_loaded = load_model()

        metrics = get_metrics(model_loaded, dataset_test=dataset_test)

        output = predict_img(
            img_path_to_detect,
            [-1, 48, 48, 3],
            model_loaded,
            class_index
        )

    except:
        return 'error', '_'

    return metrics, output


metrics, predictions = runner(
    img_path_to_detect="../../data/Datasets/fer_images/fer2013/validation/Disgust/0.jpg",
    dataset_train_path="../../data/Datasets/fer_images/fer2013/train",
    dataset_test_path="../../data/Datasets/fer_images/fer2013/validation",
    train=True
)
