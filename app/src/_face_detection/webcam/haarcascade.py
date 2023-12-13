import cv2 as cv

# read weights from data file
face_detection = cv.CascadeClassifier(
    'data/Cascades/haarcascade_frontalface_default.xml')


# start cam capturing
cam_capture = cv.VideoCapture(0)


while True:
    # read frames webcam
    ok, frame = cam_capture.read()

    # set gray scale
    imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detection
    detec = face_detection.detectMultiScale(imgGray, minSize=(100, 100))

    # result
    for x, y, w, h in detec:
        print(w, h)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv.imshow("video", frame)

    # wait for 'q' in keypress
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# memory clean
cam_capture.release()
cv.destroyAllWindows()
