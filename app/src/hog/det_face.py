import cv2 as cv
import dlib

# read image of database: (537, 1023, 3) pixels
image = cv.imread('data/Images/people2.jpg')
print(image.shape)

# detector
detector_face_hog = dlib.get_frontal_face_detector()  # type: ignore

"""
image: image to be detected
scale: image scale

"""
detec = detector_face_hog(image, 1)
print(detec)

for face in detec:
    print(face.left())
    print(face.top())
    print(face.right())
    print(face.bottom())
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv.rectangle(image, (l, t), (r, b), (0, 255, 0), 0)

cv.imshow("img", image)
cv.waitKey(0)
cv.destroyAllWindows()
