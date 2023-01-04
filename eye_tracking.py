import cv2
import numpy as np
import dlib
import math
import time


font = cv2.FONT_HERSHEY_SIMPLEX
CAMERA_SOURCE = 0
BLINK_LIMIT = 5.2  # should be changed
HEIGHT = 1280
WIDTH = 720
blinked = False
blinkedMinValue = 2

cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(3, HEIGHT)
cap.set(4, WIDTH)
# opencv cascade can also be used here just figure out the masking
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(
                                    eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(
                                    eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(
                                    eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(
                                    eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    cv2.imshow("threshold ", threshold_eye)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


def get_gaze_ratio_updown(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(
                                    eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(
                                    eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(
                                    eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(
                                    eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    up_side_threshold = threshold_eye[0: int(height/2), 0:width]
    up_side_white = cv2.countNonZero(up_side_threshold)

    down_side_threshold = threshold_eye[int(height/2): height, 0: width]
    down_side_white = cv2.countNonZero(down_side_threshold)

    cv2.imshow("threshold ", threshold_eye)
    cv2.imshow("up ", up_side_threshold)
    cv2.imshow("down", down_side_threshold)

    # print(up_side_white)
    # print(down_side_white)
    if up_side_white == 0:
        gaze_ratio = 1
    elif down_side_white == 0:
        gaze_ratio = 1
    else:
        gaze_ratio = up_side_white / down_side_white

    print(gaze_ratio)
    if gaze_ratio == 1:
        pass

    elif 0.95 >= gaze_ratio >= 0.25:
        cv2.putText(frame, "Straight ", (50, 400), font, 2, (0, 0, 255), 3)

    elif gaze_ratio > 0.95:
        cv2.putText(frame, "Down "+str("%.2f" % gaze_ratio),
                    (400, 400), font, 2, (0, 255, 0), 3)

    elif(gaze_ratio < 0.25):
        cv2.putText(frame, "UP "+str("%.2f" % gaze_ratio),
                    (800, 400), font, 2, (255, 0, 0), 3)

    return gaze_ratio


def blink_detection(markpoints, landmarks):
    left_point = (landmarks.part(
        markpoints[0]).x, landmarks.part(markpoints[0]).y)
    right_point = (landmarks.part(
        markpoints[3]).x, landmarks.part(markpoints[3]).y)
    center_top = midpoint(landmarks.part(
        markpoints[1]), landmarks.part(markpoints[2]))
    center_bottom = midpoint(landmarks.part(
        markpoints[4]), landmarks.part(markpoints[5]))

    cv2.line(frame, left_point, right_point, (203, 242, 61), 1)
    hor_line_len = math.sqrt(
        (left_point[0] - right_point[0]) ** 2 + (left_point[1] - right_point[1]) ** 2)
    cv2.line(frame, center_top, center_bottom, (203, 242, 61), 1)
    ver_line_len = math.sqrt(
        (center_top[0] - center_bottom[0]) ** 2 + (center_top[1] - center_bottom[1]) ** 2)
    ratio = hor_line_len / ver_line_len

    return ratio


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
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
        left_eye_ratio = blink_detection([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = blink_detection([42, 43, 44, 45, 46, 47], landmarks)
        # print(hor_line_len,ver_line_len)
        if (left_eye_ratio + right_eye_ratio) / 2 > BLINK_LIMIT:
            # t0 = time.time() +str(float(t0)-float(time.time())) to detect for how long the subject has eyes closed
            cv2.putText(frame, "Blinked ", (50, 300),
                        font, 2, (255, 255, 255), 2)

            if not blinked:
                blinked = True
                blinkedSec = time.time()
            if abs(time.time() - blinkedSec) > blinkedMinValue:
                # double blink add 0.350-0.700 seconds(komut7)
                blinkedSec = time.time()
                print("komut6")
        else:
            blinkedSec = time.time()
            blinked = False

        gaze_ratio_left_eye = get_gaze_ratio(
            [36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio(
            [42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        gaze_ratio_right_eye_updown = get_gaze_ratio_updown(
            [36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_left_eye_updown = get_gaze_ratio_updown(
            [42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        if gaze_ratio <= 0.65:
            cv2.putText(frame, "RIGHT "+str("%.2f" % gaze_ratio),
                        (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif 0.65 < gaze_ratio <= 1.8:
            cv2.putText(frame, "CENTER "+str("%.2f" % gaze_ratio),
                        (50, 100), font, 2, (0, 0, 255), 3)

        elif 1.8 < gaze_ratio <= 5:
            new_frame[:] = (255, 0, 0)
            cv2.putText(frame, "LEFT "+str("%.2f" % gaze_ratio),
                        (50, 100), font, 2, (0, 0, 255), 3)
        else:
            pass

    cv2.imshow("Footage", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
