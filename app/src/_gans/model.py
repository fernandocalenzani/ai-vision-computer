import tensorflow as tf


def create_generator():
    builder = tf.keras.layers

    network = tf.keras.Sequential()

    network.add(builder.Dense(
        (7*7*256),
        use_bias=False,
        input_shape=(100,))
    )
    network.add(builder.BatchNormalization())
    network.add(builder.LeakyReLU())

    network.add(builder.Reshape((7, 7, 256)))

    network.add(builder.Conv2DTranspose(
        128, (5, 5), padding='same', use_bias=False))
    network.add(builder.BatchNormalization())
    network.add(builder.LeakyReLU())

    network.add(builder.Conv2DTranspose(
        64, (5, 5), padding='same', use_bias=False, strides=(2, 2)))
    network.add(builder.BatchNormalization())
    network.add(builder.LeakyReLU())

    network.add(builder.Conv2DTranspose(
        1, (5, 5), padding='same', use_bias=False, strides=(2, 2), activation='tanh'))
    network.add(builder.BatchNormalization())
    network.add(builder.LeakyReLU())

    return network


def create_discriminator():

    builder = tf.keras.layers

    network = tf.keras.Sequential()

    network.add(builder.Conv2D(
        64, (5, 5), padding='same', strides=(2, 2), input_shape=(28, 28, 1)))
    network.add(builder.LeakyReLU())
    network.add(builder.Dropout(0.3))

    network.add(builder.Conv2D(
        128, (5, 5), padding='same', strides=(2, 2)))
    network.add(builder.LeakyReLU())
    network.add(builder.Dropout(0.3))

    network.add(builder.Flatten())

    network.add(builder.Dense(1))

    return network
