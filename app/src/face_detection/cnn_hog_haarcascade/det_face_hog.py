import cv2 as cv
import dlib

image = cv.imread('data/Images/people3.jpg')

"""----------------------------------
HOG
----------------------------------"""
detector_face_hog = dlib.get_frontal_face_detector()  # type: ignore
detec = detector_face_hog(image, 4)
for face in detec:
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv.rectangle(image, (l, t), (r, b), (0, 255, 0), 1)

cv.imshow("imghog", image)
cv.waitKey(0)
cv.destroyAllWindows()
