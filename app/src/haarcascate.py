import cv2 as cv

image = cv.resize(cv.imread('data/Images/people1.jpg'), (800, 600))
imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
face_detection = cv.CascadeClassifier(
    'data/Cascades/haarcascade_frontalface_default.xml')

# detection = [x_coord_face, y_coord_face, x_face_length, y_face_length]
detection = face_detection.detectMultiScale(imageGray, scaleFactor=)
print(detection)

for x,y,w,h in detection:
  cv.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 5)


cv.imshow("img",image)
cv.waitKey(0)
cv.destroyAllWindows()
