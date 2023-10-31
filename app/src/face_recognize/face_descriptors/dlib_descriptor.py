import os

import cv2
import dlib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

"""
PRE-PROCESSING
"""
path_weights_shape = 'data/Weights/shape_predictor_68_face_landmarks.dat'
path_weights_model = 'data/Weights/dlib_face_recognition_resnet_model_v1.dat'

path_dataset_train = 'data/Datasets/yalefaces/train'
path_dataset_test = 'data/Datasets/yalefaces/test'


def getSubjectIndex(index):
    return int(os.path.split(index)[
        1].split('.')[0].replace('subject', ''))


face_detector = dlib.get_frontal_face_detector()

point_detector = dlib.shape_predictor(path_weights_shape)

face_descriptor_extractor = dlib.face_recognition_model_v1(path_weights_model)

index = {}
idx = 0
face_descriptors = None
paths = [os.path.join(path_dataset_train, f)
         for f in os.listdir(path_dataset_train)]

"""
FACE DESCRIPTIONS
"""
for path in paths:
    image = Image.open(path).convert('RGB')
    image_np = np.array(image, 'uint8')

    detec = face_detector(image_np, 1)
    for face in detec:
        l, t, r, b = face.left(),  face.top(), face.right(), face.bottom()
        cv2.rectangle(image_np, (l, t), (r, b), (0, 255, 0), 1)

        points = point_detector(image_np, face)

        for point in points.parts():
            cv2.circle(image_np, (point.x, point.y), 2, (0, 255, 255), 1)

        face_descriptor = face_descriptor_extractor.compute_face_descriptor(
            image_np, points)
        face_descriptor = [f for f in face_descriptor]
        face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
        face_descriptor = face_descriptor[np.newaxis, :]

        if (face_descriptors is None):
            face_descriptors = face_descriptor
        else:
            face_descriptors = np.concatenate((
                face_descriptors, face_descriptor), axis=0)

    index[idx] = path
    idx += 1

"""
TEST DISTANCE BETWEEN FACES
"""
face_descriptors[131]

# calculate the distance between vectors or similarities
# if sim --> 0 = more similar

sim = np.linalg.norm(face_descriptors[131] - face_descriptors[1:], axis=1)
less_d = np.argmin(sim)


"""
FACE DETECTION
"""
min_confidance = 0.5
predictions = []
exp_output = []

paths = [os.path.join(path_dataset_test, f)
         for f in os.listdir(path_dataset_test)]

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
            pred_name = getSubjectIndex(index[min_index])
        else:
            pred_name = None

        real_name = getSubjectIndex(index[min_index])

        predictions.append(pred_name)
        exp_output.append(real_name)

        cv2.putText(image, 'Pred: ' + str(pred_name), (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        cv2.putText(image, 'Exp: ' + str(real_name), (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

        cv2.imshow("img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


"""
PERFORMANCE
"""
predictions = np.array(predictions)
exp_output = np.array(exp_output)

acc = accuracy_score(exp_output, predictions)
