import cv2
import  numpy as np
import dlib
from math import hypot


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Setting the font type
font = cv2.FONT_HERSHEY_COMPLEX


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


def get_gazing_ratio(eye_points, facial_landmarks):

    # Getting  coordinates of landmark points of the eye
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x,facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    # mask
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    # Separating the eye_frame
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])

    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_scale_eye = eye[min_y: max_y, min_x: max_x]
    # grayscale_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

    # Take the  binary of the eye frame
    _, threshold_eye = cv2.threshold(gray_scale_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape

    # Separating the left eye
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    # Counting white spaces of the left eye(Sclera)
    left_side_white_spaces = cv2.countNonZero(left_side_threshold)

    # Separating the right eye
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    # Counting white spaces of the right eye(Sclera)
    right_side_white_spaces = cv2.countNonZero(right_side_threshold)

    try:
        # Calculating the gaze ratio
        gazing_ratio = left_side_white_spaces / right_side_white_spaces
        return gazing_ratio

    except ZeroDivisionError:
        return -1


while True:
    _, frame = cap.read()
    # Indicator frame
    indicator_frame = np.zeros((500, 5000, 3), np.uint8)
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

        # Get gazing ratio
        gazing_ratio_left_eye = get_gazing_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gazing_ratio_right_eye = get_gazing_ratio([42, 43, 44, 45, 46, 47], landmarks)
        average_gaze_ratio = (gazing_ratio_left_eye + gazing_ratio_right_eye) / 2
        cv2.putText(frame, str(average_gaze_ratio), (50, 450), font, 2, (255, 0, 255), 3)

        if average_gaze_ratio < 0:
            cv2.putText(frame, "BRING YOUR EYES CLOSER TO CAMERA", (50, 350), font, 2, (0, 0, 255), 3)
        elif 0 <= average_gaze_ratio <= 1:
            cv2.putText(frame, "LEFT", (50, 350), font, 2, (0, 0, 255), 3)
            indicator_frame[:] = (0, 0, 255)
        elif 1 < average_gaze_ratio < 3:
            cv2.putText(frame, "CENTER", (50, 350), font, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "RIGHT", (50, 350), font, 2, (0, 0, 255), 3)
            indicator_frame[:] = (0, 255, 0)



        # threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        # cv2.imshow("Threshold", threshold_eye)
        # cv2.imshow("Left Eye", left_eye_threshold)
        # cv2.imshow("Right Eye", right_eye_threshold)

    cv2.imshow("Frame", frame)
    cv2.imshow("Indicator Frame", indicator_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
