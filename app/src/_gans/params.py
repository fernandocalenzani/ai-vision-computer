import numpy as np
import tensorflow as tf
from model import create_discriminator, create_generator

data = {}

data['x_norm'] = 255/2
data['buffer_size'] = 60000
data['batch_size'] = 256  # mini batch grandient descent
(data['x_train'], data['y_train']), _ = tf.keras.datasets.mnist.load_data()
data['cross_entropy'] = tf.keras.losses.BinaryCrossentropy(from_logits=True)
data['epochs'] = 100
data['dim_noise'] = 100
data['n_imgs'] = 16
data['img_teste'] = np.random.normal([data['n_imgs'], data['dim_noise']])
data['network_generator'] = create_generator()
data['network_discriminator'] = create_discriminator()
data['gen_optimizer'] = tf.keras.optimizers.Adam(learning_rate=0.00001)
data['disc_optimizer'] = tf.keras.optimizers.Adam(learning_rate=0.00001)

data['x_train'] = data['x_train'].reshape(
    data['x_train'].shape[0],
    data['x_train'].shape[1],
    data['x_train'].shape[2],
    1).astype('float32')

data['x_train'] = (data['x_train'] - data['x_norm'])/(data['x_norm'])
data['x_train'] = tf.data.Dataset.from_tensor_slices(
    data['x_train']).shuffle(data['buffer_size']).batch(data['batch_size'])
