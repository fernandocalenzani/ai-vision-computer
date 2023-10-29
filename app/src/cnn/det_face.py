import cv2 as cv
import dlib

# read image of database: (537, 1023, 3) pixels
image = cv.imread('data/Images/people2.jpg')
print(image.shape)

# detector
detector_face_cnn = dlib.cnn_face_detection_model_v1(  # type: ignore
    'data/Weights/mmod_human_face_detector.dat')

"""
image: image to be detected
scale: image scale

"""
detec = detector_face_cnn(image, 1)
print(detec)

for face in detec:
    l, t, r, b, c = face.rect.left(), face.rect.top(
    ), face.rect.right(), face.rect.bottom(), face.confidence
    print(c)
    cv.rectangle(image, (l, t), (r, b), (0, 255, 0), 1)


cv.imshow("img", image)
cv.waitKey(0)
cv.destroyAllWindows()
