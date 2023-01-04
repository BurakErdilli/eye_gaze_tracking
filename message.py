import cv2
import numpy as np
import dlib
import math
import time
#clock.tick eklenecek, kapalı goz için timer tutulacak, double blink tespit edilecek
# presets
font = cv2.FONT_HERSHEY_SIMPLEX
CAMERA_SOURCE = 0
BLINK_LIMIT = 5.2 #should be changed
HEIGHT = 1280
WIDTH = 720
cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(3, HEIGHT)
cap.set(4, WIDTH)
# opencv cascade can also be used here just figure out the masking
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def blink_detection(markpoints, landmarks):
    left_point = (landmarks.part(markpoints[0]).x, landmarks.part(markpoints[0]).y)
    right_point = (landmarks.part(markpoints[3]).x, landmarks.part(markpoints[3]).y)
    center_top = midpoint(landmarks.part(markpoints[1]), landmarks.part(markpoints[2]))
    center_bottom = midpoint(landmarks.part(markpoints[4]), landmarks.part(markpoints[5]))

    cv2.line(frame, left_point, right_point, (203, 242, 61), 1)
    hor_line_len = math.sqrt((left_point[0] - right_point[0]) ** 2 + (left_point[1] - right_point[1]) ** 2)
    cv2.line(frame, center_top, center_bottom, (203, 242, 61), 1)
    ver_line_len = math.sqrt((center_top[0] - center_bottom[0]) ** 2 + (center_top[1] - center_bottom[1]) ** 2)
    ratio = hor_line_len / ver_line_len

    return ratio


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        # marking faces
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (216, 255, 158), 1)
        # cv2.circle(frame, ((x + x1) // 2, (y + y1) // 2), (y + x) // 4, (216, 255, 158), 3)
        # grayscale marking

        landmarks = predictor(gray, face)
        # blink detection
        right_eye_ratio = blink_detection([36, 37, 38, 39, 40, 41], landmarks)
        left_eye_ratio = blink_detection([42, 43, 44, 45, 46, 47], landmarks)
        # print(hor_line_len,ver_line_len)
        if (left_eye_ratio + right_eye_ratio) / 2 > BLINK_LIMIT:
            # t0 = time.time() +str(float(t0)-float(time.time())) to detect for how long the subject has eyes closed
            cv2.putText(frame, "blink ;) ", (50,50), font, 2, (255, 255, 255),2)

        right_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                     (landmarks.part(37).x, landmarks.part(37).y),
                                     (landmarks.part(38).x, landmarks.part(38).y),
                                     (landmarks.part(39).x, landmarks.part(39).y),
                                     (landmarks.part(40).x, landmarks.part(40).y),
                                     (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        left_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                    (landmarks.part(43).x, landmarks.part(43).y),
                                    (landmarks.part(44).x, landmarks.part(44).y),
                                    (landmarks.part(45).x, landmarks.part(45).y),
                                    (landmarks.part(46).x, landmarks.part(46).y),
                                    (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(frame, [right_eye_region], True, (0, 0, 255), 2)
        cv2.polylines(mask, [right_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [right_eye_region], 255)
        right_eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(right_eye_region[:, 0])
        max_x = np.max(right_eye_region[:, 0])
        min_y = np.min(right_eye_region[:, 1])
        max_y = np.max(right_eye_region[:, 1])
        gray_eye = right_eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        height, width = threshold_eye.shape

        total_white = cv2.countNonZero(threshold_eye)

        right_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        right_side_white = cv2.countNonZero(right_side_threshold)

        left_side_threshold = threshold_eye[0: height, int(width / 2): width]
        left_side_white = cv2.countNonZero(left_side_threshold)

        try:    
            l_f_ratio=left_side_white/right_side_white   
            print(l_f_ratio)
            cv2.putText(frame, str(l_f_ratio), (50, 50), font, 2, (255, 255, 255), 3)
            if l_f_ratio < 1:
                cv2.putText(frame, "", (50, 300), font, 2, (255, 255, 255), 3)
            elif l_f_ratio > 3:
                cv2.putText(frame, "right", (50, 300), font, 2, (255, 255, 255), 3)
                


        except ZeroDivisionError:
            print("divby zero \n")



        cv2.putText(frame, str(left_side_white), (50, 100), font, 2, (255, 0, 255), 3)
        cv2.putText(frame, str(right_side_white), (50, 150), font, 2, (255, 0, 255), 3)
        cv2.putText(frame, str(total_white), (50, 200), font, 2, (255, 0, 255), 3)

        eye = cv2.resize(gray_eye, None, fx=5, fy=5)
        cv2.imshow("left", left_side_threshold)
        cv2.imshow("right", right_side_threshold)
        cv2.imshow("Threshold", threshold_eye)

    cv2.imshow("Footage", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()