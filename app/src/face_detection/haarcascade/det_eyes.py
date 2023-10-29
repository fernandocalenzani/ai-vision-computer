import cv2 as cv

# read image of database: (1280, 1920, 3) pixels
image = cv.imread('data/Images/people1.jpg')
print(image.shape)

# change to gray color, to decrease the information in image (600, 800) pixels:
imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
print(imageGray.shape)

# read xml data to classify the image
face_training = cv.CascadeClassifier(
    'data/Cascades/haarcascade_frontalface_default.xml')

eyes_training = cv.CascadeClassifier(
    'data/Cascades/haarcascade_eye.xml')

"""
detection: the result is the matrix = [x_coord_face, y_coord_face, x_face_length, y_face_length]

scaleFactor: scaleFactor should largest 1. The algorithm increases the size of the image, that is, it is good for detecting small objects. Otherwise, if you decrease the scaleValue value, the algorithm decreases the image size, good for detecting large objects

minNeighboors: minimum number of neighbors to consider for each face in the image

minSize: minimum size of the face to detect faces
maxSize: maximum size of the face to detect faces
"""
face_detec = face_training.detectMultiScale(
    imageGray, scaleFactor=1.3, minSize=(30, 30))
print(face_detec)
for x, y, w, h in face_detec:
    print(w,h)
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

eyes_detec = eyes_training.detectMultiScale(
    imageGray, scaleFactor=1.1, minNeighbors=10, maxSize=(70,70))
print(eyes_detec)
for x, y, w, h in eyes_detec:
    print(w, h)
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)


cv.imshow("img", image)
cv.waitKey(0)
cv.destroyAllWindows()
