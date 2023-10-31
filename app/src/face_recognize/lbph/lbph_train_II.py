import os

import cv2
import numpy as np
from PIL import Image

"""
PRE-PROCESSING
"""
db_path = 'data/Datasets/jones_gabriel/jones_gabriel'

# Get image data
def get_img_data():
    # getting image paths
    paths = [os.path.join(db_path, f)
             for f in os.listdir(os.path.join(db_path))]
    faces = []
    ids = []

    # getting each path
    for path in paths:
        # transforming to gray
        image = np.array(Image.open(path).convert('L'), 'uint8')

        id = int(path.split('.')[1])
        ids.append(id)

        faces.append(image)

    return np.array(ids), faces


ids, faces = get_img_data()

"""
TRAINING AND CLASSIFICATION
LBPH PARAMETERS:
radius: pixels range
neighbors: number of pixels used in condition
grid_X and grid_Y: grade length (lin, col)
threshold: confidence level
"""
lbph_classifier = cv2.face_LBPHFaceRecognizer.create(
    radius=4, neighbors=14, grid_x=9, grid_y=9)
lbph_classifier.train(faces, ids)
lbph_classifier.write('src/face_recognize/webcam/lbph_classifier_II.yml')
