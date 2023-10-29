import cv2 as cv
import dlib

image = cv.imread('data/Images/people3.jpg')

"""----------------------------------
CNN
----------------------------------"""

detector_face_cnn = dlib.cnn_face_detection_model_v1(  # type: ignore
    'data/Weights/mmod_human_face_detector.dat')

detec = detector_face_cnn(image, 4)

for face in detec:
    l, t, r, b, c = face.rect.left(), face.rect.top(
    ), face.rect.right(), face.rect.bottom(), face.confidence
    cv.rectangle(image, (l, t), (r, b), (0, 0, 255), 1)

cv.imshow("imgcnn", image)
cv.waitKey(0)
cv.destroyAllWindows()
