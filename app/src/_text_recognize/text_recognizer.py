import cv2
import numpy as np
import pytesseract
import seaborn as sns
import tensorflow as tf
from keras.models import save_model
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


def img2str(img, lang, config_tesseract):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return pytesseract.image_to_string(image=img, lang=lang, config=config_tesseract)


def get_languages():
    return pytesseract.get_languages()


def runner(img, lang, config_tesseract):
    return img2str(img, lang=lang, config_tesseract=config_tesseract)


text = runner(
    img="../../data/Images/saida.jpg",
    lang='por',
    config_tesseract='--tessdata-dir ./configs --psm 7'
)
