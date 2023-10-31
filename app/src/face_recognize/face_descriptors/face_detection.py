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

face_descriptor_extractor = dlib.face_recognition_model_v1(
    'data/Weights/dlib_face_recognition_resnet_model_v1.dat')

index = {}
idx = 0
face_descriptors = None

"""
FACE DETECTION
"""
min_confidance = 0.5

paths = [os.path.join('data/Datasets/yalefaces/test', f)
         for f in os.listdir('data/Datasets/yalefaces/test')]

for path in paths:
    image = np.array(Image.open(path).convert('RGB'), 'uint8')

    detections = face_detector(image, 1)

    for face in detections:
        points = point_detector(image, face)
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(
            image, points)

        # converting to list
        face_descriptor = [f for f in face_descriptor]

        # transforming to np array
        face_descriptor = np.asarray(face_descriptor, dtype=np.float64)

        # add new axis
        face_descriptor = face_descriptor[np.newaxis, :]

        distance = np.linalg.norm(face_descriptor - face_descriptors, axis=1)
        min_index = np.argmin(distance)
        min_distance = distance[min_index]

        if (min_distance <= min_confidance):
            pred_name = int(os.path.split(index[min_index])[
                            1].split('.')[0].replace('subject', ''))
        else:
            pred_name = None

        real_name = int(os.path.split(index[min_index])[
            1].split('.')[0].replace('subject', ''))

        cv2.putText(image, 'Pred: ' + str(pred_name), (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        cv2.putText(image, 'Exp: ' + str(real_name), (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

        cv2.imshow("img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
