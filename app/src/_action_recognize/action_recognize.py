import cv2
import matplotlib.pyplot as plt


def preprocessing(frame, scalefactor=1.0/255):
    img_blob = cv2.dnn.blobFromImage(
        frame, scalefactor=scalefactor, size=(frame.shape[1], frame.shape[0]))

    return img_blob, frame


def load_network(path_network_config, path_weights):
    network = cv2.dnn.readNetFromCaffe(path_network_config, path_weights)
    return network


def actions_prediction(points):
    head, hand_left, hand_right = 0, 0, 0

    for i, points in enumerate(points):
        if i == 0:
            head = points[1]
        elif i == 4:
            hand_right = points[1]
        elif i == 7:
            hand_left = points[1]

    if hand_right <= head and hand_left <= head:
        return True
    else:
        return False


def run_frames(
    video_path,
    scaleFactor,
    path_network_config,
    path_weights,
    n_pointers,
    threshold
):
    points = []
    conection_points = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [
        6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

    cap = cv2.VideoCapture(video_path)
    connected, frame = cap.read()

    save_video = cv2.VideoWriter('../../data/Videos/emotion_test01_result.avi',
                                 cv2.VideoWriter_fourcc(*'XVID'),
                                 10,
                                 (frame.shape[1], frame.shape[0]))

    while cv2.waitKey(1) < 0:
        connected, frame = cap.read()

        if not connected:
            break

        img_blob, frame = preprocessing(
            frame=frame, scalefactor=scaleFactor)
        network = load_network(path_network_config, path_weights)

        network.setInput(img_blob)

        output = network.forward()

        pos_l = output.shape[3]
        pos_h = output.shape[2]

        for i in range(n_pointers):
            confidance_map = output[0, i, :, :]
            _, confidance, _, point = cv2.minMaxLoc(confidance_map)

            x = int((img_blob.shape[3] * point[0])/pos_l)
            y = int((img_blob.shape[2] * point[1])/pos_h)

            if confidance > threshold:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), thickness=-1)
                cv2.putText(frame, '{}'.format(i), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

                points.append((x, y))
            else:
                points.append(None)

        for conection in conection_points:
            partA = conection[0]
            partB = conection[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (255, 0, 0))

        if actions_prediction(points) == True:
            cv2.putText(frame, 'OK ', (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255))

        save_video.write(frame)
        save_video.release()

    return output, points


output, points = run_frames(
    video_path='../../data/Videos/gesture1.mp4',
    scaleFactor=1.0/255,
    path_network_config='../../data/Weights/pose_deploy_linevec_faster_4_stages.prototxt',
    path_weights='../../data/Weights/pose_iter_160000.caffemodel',
    n_pointers=15,
    threshold=0.01
)
