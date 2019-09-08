import cv2
import numpy as np
import dlib 
import pyglet
from math import hypot
import time

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


# Text board
display_board = np.zeros((500, 500), np.uint8)
display_board[:] = 255


# Key board Settings
keyboard = np.zeros((1000, 1000, 3), np.uint8)

letter_set_1 = {0: "Q", 1: "W", 2: "E",  3: "R", 4: "T",
                5: "Y", 6: "U", 7: "I",  8: "O", 9: "P",
                10: "A", 11: "S", 12: "D",  13: "F", 14: "G",
                15: "H", 16: "J", 17: "K",  18: "L", 19: "Z",
                20: "X", 21: "C", 22: "V",  23: "B", 24: "N",
                25: "M"}


def generate_key(x, y, letter_index, highlight_letter):

    key_height = 200
    key_width = 200
    key_thickness = 3

    if highlight_letter:
        cv2.rectangle(keyboard, (x + key_thickness, y + key_thickness),
                      (x + key_width - key_thickness, y + key_width - key_thickness), (255, 255, 255), -1)
    else:
        cv2.rectangle(keyboard, (x + key_thickness, y + key_thickness),
                      (x + key_width - key_thickness, y+key_width - key_thickness), (255, 0, 0), key_thickness)

    # Keyboard Letter Settings
    keyboard_font = cv2.FONT_HERSHEY_PLAIN
    letter_scale = 10
    letter_thickness = 4
    keyboard_letter_size = cv2.getTextSize(letter_set_1[letter_index], keyboard_font, letter_scale, letter_thickness)[0]
    letter_width, letter_height = keyboard_letter_size[0], keyboard_letter_size[1]
    letter_x = int((key_width - letter_width) / 2) + x
    letter_y = int((key_height + letter_height) / 2) + y
    cv2.putText(keyboard, letter_set_1[letter_index], (letter_x, letter_y), keyboard_font, letter_scale, (255, 0, 0),
                letter_thickness)


# Sound
success_notification = pyglet.media.load("sound.wav", streaming=False)
left_notification = pyglet.media.load("left.wav", streaming=False)
right_notification = pyglet.media.load("right.wav", streaming=False)

# Counting the frame
frame_counter = 0
blinking_frame_counter = 0
letter_frame_index = 0
typed_text = ""
eyes_position_current = "left"
eyes_position_previous = "left"
# Main Program
while True:
    _, frame = cap.read()
    keyboard[:] = (0, 0, 0)
    frame_counter += 1
    # Indicator frame
    indicator_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    selected_letter = letter_set_1[letter_frame_index]

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
            blinking_frame_counter += 1
            frame_counter -= 1

            if blinking_frame_counter == 5:
                typed_text += selected_letter
                success_notification.play()
                time.sleep(1)
        else:
            blinking_frame_counter = 0

        # Get gazing ratio
        gazing_ratio_left_eye = get_gazing_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gazing_ratio_right_eye = get_gazing_ratio([42, 43, 44, 45, 46, 47], landmarks)
        average_gaze_ratio = (gazing_ratio_left_eye + gazing_ratio_right_eye) / 2
        cv2.putText(frame, str(average_gaze_ratio), (50, 450), font, 2, (255, 0, 255), 3)

        if average_gaze_ratio < 0:
            cv2.putText(frame, "BRING YOUR EYES CLOSER TO CAMERA", (50, 350), font, 2, (0, 0, 255), 3)
        elif 0 <= average_gaze_ratio <= 0.9:
            cv2.putText(frame, "LEFT", (50, 350), font, 2, (0, 0, 255), 3)
            #indicator_frame[:] = (0, 0, 255)
            eyes_position_current = "left"
            if eyes_position_current != eyes_position_previous:
                # left_notification.play()
                # time.sleep(1)
                print("left")
                eyes_position_previous = "left"

        elif 0.9 < average_gaze_ratio < 3:
            cv2.putText(frame, "CENTER", (50, 350), font, 2, (0, 0, 255), 3)

        else:
            cv2.putText(frame, "RIGHT", (50, 350), font, 2, (0, 0, 255), 3)
            # indicator_frame[:] = (0, 255, 0)
            eyes_position_current = "right"
            if eyes_position_current != eyes_position_previous:
                # right_notification.play()
                # time.sleep(1)
                print("right")
                eyes_position_previous = "right"

        # Generating keyboard
        if frame_counter == 10:
            letter_frame_index += 1
            frame_counter = 0

        if letter_frame_index == 15:
            letter_frame_index = 0

        key_x = 0
        key_y = 0
        for i in range(15):
            if i == letter_frame_index:
                generate_key(key_x, key_y, letter_frame_index, True)
            else:
                generate_key(key_x, key_y, i, False)

            if (i + 1) % 5 == 0:
                key_x = 0
                key_y = key_y + 200
            else:
                key_x = key_x + 200
        cv2.putText(display_board, typed_text, (10, 100), font, 4, 0, 3)

    cv2.imshow("Frame", frame)
    #cv2.imshow("Indicator Frame", indicator_frame)
    cv2.imshow("Virtual Keyboard", keyboard)
    cv2.imshow("Display board", display_board)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
