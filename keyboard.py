import cv2
import numpy as np

keyboard = np.zeros((1000, 1500, 3), np.uint8)


# drawing keys
def generate_key(x, y, letter):

    key_height = 200
    key_width = 200
    key_thickness = 3
    cv2.rectangle(keyboard, (x + key_thickness, y + key_thickness),
                  (x + key_width - key_thickness, y+key_width - key_thickness), (255, 0, 0), key_thickness)

    # Keyboard Letter Settings
    keyboard_font = cv2.FONT_HERSHEY_PLAIN
    letter_scale = 10
    letter_thickness = 4
    keyboard_letter_size = cv2.getTextSize(letter, keyboard_font, letter_scale, letter_thickness)[0]
    letter_width, letter_height = keyboard_letter_size[0], keyboard_letter_size[1]
    letter_x = int((key_width - letter_width) / 2) + x
    letter_y = int((key_height + letter_height) / 2) + y
    cv2.putText(keyboard, letter, (letter_x, letter_y), keyboard_font, letter_scale, (255, 0, 0), letter_thickness)


generate_key(0, 0, "A")
generate_key(200, 0, "B")
generate_key(400, 0, "C")

cv2.imshow("Virtual Keyboard", keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
