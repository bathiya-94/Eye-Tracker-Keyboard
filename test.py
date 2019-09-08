import pyglet
import time

success_notification = pyglet.media.load("slow-spring-board.wav")
left_notification = pyglet.media.load("left.wav", streaming="False")
right_notification = pyglet.media.load("right.wav", streaming="False")
success_notification.play()

time.sleep(0.5)

# left_notification.play()
# time.sleep(.500)
# right_notification.play()
# time.sleep(.500)

