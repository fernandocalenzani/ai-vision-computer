import cv2 as cv

# read image of database: (1280, 1920, 3) pixels
image = cv.imread('data/Images/people1.jpg')
print(image.shape)

# resize to # read image of database: (800, 600, 3) pixels to decrease the processing time
imageResized = cv.resize(image, (800, 600))
print(imageResized.shape)

# change to gray color, to decrease the information in image (600, 800) pixels:
imageGray = cv.cvtColor(imageResized, cv.COLOR_BGR2GRAY)
print(imageGray.shape)

# read xml data to classify the image
face_detection = cv.CascadeClassifier(
    'data/Cascades/haarcascade_frontalface_default.xml')

"""
detection: the result is the matrix = [x_coord_face, y_coord_face, x_face_length, y_face_length]

scaleFactor: scaleFactor should largest 1. The algorithm increases the size of the image, that is, it is good for detecting small objects. Otherwise, if you decrease the scaleValue value, the algorithm decreases the image size, good for detecting large objects
"""
detection = face_detection.detectMultiScale(imageGray, scaleFactor=1.09)
print(detection)


# Plotting detections
for x, y, w, h in detection:
    cv.rectangle(imageResized, (x, y), (x+w, y+h), (0, 255, 0), 1)

cv.imshow("img", imageResized)
cv.waitKey(0)
cv.destroyAllWindows()
