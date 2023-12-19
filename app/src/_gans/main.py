from gans import train_gan
from params import data

train_gan(dataset=data['x_train'],
          epochs=data['epochs'], imgs_test=data['y_train'])
