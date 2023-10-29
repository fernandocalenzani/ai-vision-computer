import cv2 as cv

# read image of database: (537, 1023, 3) pixels
image = cv.imread('data/Images/people2.jpg')
print(image.shape)

# change to gray color, to decrease the information in image (537, 1023) pixels:
imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
print(imageGray.shape)

# read xml data to classify the image
face_detection = cv.CascadeClassifier(
    'data/Cascades/haarcascade_frontalface_default.xml')

"""
detection: the result is the matrix = [x_coord_face, y_coord_face, x_face_length, y_face_length]

scaleFactor: scaleFactor should largest 1. The algorithm increases the size of the image, that is, it is good for detecting small objects. Otherwise, if you decrease the scaleValue value, the algorithm decreases the image size, good for detecting large objects

minNeighboors: minimum number of neighbors to consider for each face in the image

minSize: minimum size of the face to detect faces
maxSize: maximum size of the face to detect faces
"""

detection = face_detection.detectMultiScale(
    imageGray, scaleFactor=1.2, minNeighbors=3, minSize=(28, 28), maxSize=(100,100))

# Plotting detections
for x, y, w, h in detection:
    print(w,h)
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

cv.imshow("img", image)
cv.waitKey(0)
cv.destroyAllWindows()
