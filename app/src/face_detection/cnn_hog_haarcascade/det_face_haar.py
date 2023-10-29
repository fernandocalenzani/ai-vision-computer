import cv2 as cv

image = cv.imread('data/Images/people3.jpg')

"""----------------------------------
HAARCASCADE
----------------------------------"""
imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
face_detection = cv.CascadeClassifier(
    'data/Cascades/haarcascade_frontalface_default.xml')

detection = face_detection.detectMultiScale(
    imageGray, scaleFactor=1.001, minNeighbors=5, minSize=(5, 5))

for x, y, w, h in detection:
    print(w, h)
    cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)

cv.imshow("imghaar", image)
cv.waitKey(0)
cv.destroyAllWindows()
