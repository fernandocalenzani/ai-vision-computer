import cv2
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.models import save_model
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

cap = cv2.VideoCapture('../../data/Videos/emotion_test01_result.avi')

connected, video = cap.read()

