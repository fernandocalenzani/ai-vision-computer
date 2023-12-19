import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from params import data


def train_gan(dataset, epochs, imgs_test):
    for epoch in range(epochs):
        for img_batch in dataset:
            training(img_batch)

        print('Epoch: ', epoch + 1)
        img_generated = data['network_generator'](imgs_test, training=False)

        fig = plt.figure(figsize=(10, 10))

        for i in range(img_generated.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(img_generated[i, :, :, 0] *
                       data['x_norm'] + data['x_norm'], cmap='gray')
            plt.axis('off')
            plt.show()

    return


def discriminator_losses(expected_output, fake_output):
    real_loss = data['cross_entropy'](
        tf.ones_like(expected_output), expected_output)
    fake_loss = data['cross_entropy'](tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss


def generator_losses(fake_output):
    return data['cross_entropy'](tf.ones_like(fake_output), fake_output)


@tf.function
def training(imgs):
    noise = tf.random.normal([data['batch_size'], data['dim_noise']])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        imgs_gen = data['network_generator'](noise, training=True)

        expected_outputs = data['network_discriminator'](imgs, training=True)
        fake_outputs = data['network_discriminator'](imgs_gen, training=True)

        loss_gen = generator_losses(fake_outputs)
        loss_disc = discriminator_losses(expected_outputs, fake_outputs)

    gradient_gen = gen_tape.gradient(
        loss_gen, data['network_generator'].trainable_variables)
    gradient_disc = disc_tape.gradient(
        loss_disc, data['network_discriminator'].trainable_variables)

    data['gen_optimizer'].apply_gradients(
        zip(gradient_gen, data['network_generator'].trainable_variables))

    data['disc_optimizer'].apply_gradients(
        zip(gradient_disc, data['network_discriminator'].trainable_variables))
