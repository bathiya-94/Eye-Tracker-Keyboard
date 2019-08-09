import cv2
import numpy as np

keyboard = np.zeros((1000, 1000, 3), np.uint8)

letter_set_1 = {0: "Q", 1: "W", 2: "E",  3: "R", 4: "T",
                5: "Y", 6: "U", 7: "I",  8: "O", 9: "P",
                10: "A", 11: "S", 12: "D",  13: "F", 14: "G",
                15: "H", 16: "J", 17: "K",  18: "L", 19: "Z",
                20: "X", 21: "C", 22: "V",  23: "B", 24: "N",
                25: "M"}


# drawing keys
def generate_key(x, y, letter_index):

    key_height = 200
    key_width = 200
    key_thickness = 3
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


key_x = 0
key_y = 0
for i in range(26):
    print(i)
    generate_key(key_x, key_y, i)

    if (i + 1) % 5 == 0:
        key_x = 0
        key_y = key_y + 200
    else:
        key_x = key_x + 200


cv2.imshow("Virtual Keyboard", keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
