import os

import cv2
import dlib
import numpy as np
from PIL import Image

"""
PRE-PROCESSING
"""

face_detector = dlib.get_frontal_face_detector()

point_detector = dlib.shape_predictor(
    'data/Weights/shape_predictor_68_face_landmarks.dat')

image = cv2.imread('data/Images/people2.jpg')

detec = face_detector(image, 1)
print(detec)

for face in detec:
    points = point_detector(image, face)
    for point in points.parts():
        cv2.circle(image, (point.x, point.y), 2, (0, 0, 255), 1)

    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 1)

cv2.imshow("img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
