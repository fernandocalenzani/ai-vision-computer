import cv2

detector_face = cv2.CascadeClassifier(
    "data/Cascades/haarcascade_frontalface_default.xml")

face_classifier = cv2.face.LBPHFaceRecognizer_create()

face_classifier.read('src/face_recognize/webcam/lbph_classifier_II.yml')

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
cam = cv2.VideoCapture(0)

while True:
    ok, frame = cam.read()
    print(ok, frame)
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = detector_face.detectMultiScale(
        image_gray, scaleFactor=1.5, minSize=(30, 30))

    for (x, y, w, h) in detections:
        image_face = cv2.resize(image_gray[y:y+w, x:x+h], (220, 220))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        id, conf = face_classifier.predict(image_face)

        name = ""

        if id == 1:
            name = "Jones"
        elif id == 2:
            name = "Gabriel"
        else:
            name = "not found user"

        cv2.putText(frame, name, (x, y + (w+30)), font, 2, (0, 0, 255))
        cv2.putText(frame, str(conf), (x, y + (h+50)), font, 1, (0, 0, 255))

    cv2.imshow("video", frame)

    # wait for 'q' in keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# memory clean
cam.release()
cv2.destroyAllWindows()
