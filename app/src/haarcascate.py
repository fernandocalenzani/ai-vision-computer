import cv2 as cv


image = cv.imread('data/Images/people1.jpg')
imageGray = cv.cvtColor(cv.resize(image, (800,600)), cv.COLOR_BGR2GRAY)
face_detection = cv.CascadeClassifier('data/Cascades/haarcascade_frontalface_default.xml')
detection = face_detection.detectMultiScale(imageGray)

print(detection)
