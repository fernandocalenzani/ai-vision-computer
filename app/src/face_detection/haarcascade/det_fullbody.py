import cv2 as cv

# read image of database: (1280, 1920, 3) pixels
image = cv.imread('data/Images/people3.jpg')
print(image.shape)

# change to gray color, to decrease the information in image (600, 800) pixels:
imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
print(imageGray.shape)

training = cv.CascadeClassifier(
    'data/Cascades/fullbody.xml')

"""
detection: the result is the matrix = [x_coord_face, y_coord_face, x_face_length, y_face_length]

scaleFactor: scaleFactor should largest 1. The algorithm increases the size of the image, that is, it is good for detecting small objects. Otherwise, if you decrease the scaleValue value, the algorithm decreases the image size, good for detecting large objects

minNeighboors: minimum number of neighbors to consider for each face in the image

minSize: minimum size of the face to detect faces
maxSize: maximum size of the face to detect faces
"""
detec = training.detectMultiScale(
    imageGray, scaleFactor=1.05, minNeighbors=5)
print(detec)
for x, y, w, h in detec:
    print(w, h)
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

cv.imshow("img", image)
cv.waitKey(0)
cv.destroyAllWindows()
