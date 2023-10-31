import os

import cv2
import numpy as np
import seaborn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()

lbph_face_classifier.read('src/face_recognize/lbph_classifier.yml')
db_path = 'data/Datasets/yalefaces/test'
paths = [os.path.join(db_path, f) for f in os.listdir(os.path.join(db_path))]

predictions = []
exp_predictions = []

# getting each path
for path in paths:
    # transforming to gray
    image = np.array(Image.open(path).convert('L'), 'uint8')
    predict, _ = lbph_face_classifier.predict(image)
    expected_predict = int(os.path.split(path)[1].split('.')[
        0].replace('subject', ''))

    predictions.append(predict)
    exp_predictions.append(expected_predict)


print(predictions)
print(exp_predictions)

acc = accuracy_score(exp_predictions, predictions)
cm = confusion_matrix(exp_predictions, predictions)
predictions = np.array(predictions)
exp_predictions = np.array(exp_predictions)
seaborn.heatmap(cm, annot=True)

print(acc)

""" cv2.putText(image_np, 'Pred: ' + str(predict[0]), (10, 30),
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.putText(image_np, 'Exp: ' + str(expected_predict), (10, 50),
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.imshow("img", image_np)

cv2.waitKey(0)
cv2.destroyAllWindows()
 """
