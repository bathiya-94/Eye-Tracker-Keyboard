import cv2
import  numpy as np
import dlib
from math import hypot


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[4]), facial_landmarks.part(eye_points[5]))

    # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # vertical_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    vertical_line_len = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    hor_line_len = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))

    return hor_line_len / vertical_line_len


font = cv2.FONT_HERSHEY_COMPLEX


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()

        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio)/2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 3, (255, 0, 0))

        # Gaze detection
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])

        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        eye_frame = frame[min_y: max_y, min_x: max_x]
        grayscale_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

        # Take the inverse binary of the eye frame
        _, threshold_eye = cv2.threshold(grayscale_eye, 70, 255, cv2.THRESH_BINARY_INV)
        eye_frame = cv2.resize(eye_frame, None, fx=5, fy=5)

        cv2.imshow("Eye", eye_frame)
        cv2.imshow("Threshold", threshold_eye)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
