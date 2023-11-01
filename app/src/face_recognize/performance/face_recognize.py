import os

import cv2
import numpy as np
import seaborn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

predictions = []
exp_outputs = []
lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()

lbph_face_classifier.read('src/face_recognize/lbph_classifier.yml')
db_path = 'data/Datasets/yalefaces/test'
paths = [os.path.join(db_path, f) for f in os.listdir(os.path.join(db_path))]

for path in paths:
    image = np.array(Image.open(path).convert('L'), 'uint8')
    prediction, _ = lbph_face_classifier.predict(image)
    exp_output = int(os.path.split(path)[1].split('.')[
        0].replace('subject', ''))

    predictions.append(predictions)
    exp_outputs.append(exp_output)

predictions = np.array(predictions)
exp_outputs = np.array(exp_outputs)

acc = accuracy_score(exp_outputs, predictions)
n_img = len(predictions)
mc = confusion_matrix(exp_outputs, predictions)
seaborn.heatmap(mc, annot=True);
