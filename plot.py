import cv2
import numpy as np
import matplotlib.pyplot as plt

video_capture = cv2.VideoCapture(1)
x = np.array([])
y = np.array([])

while True:
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_mean = np.mean(gray)
    y = np.append(y, gray_mean)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.plot(y)
plt.show()
video_capture.release()
cv2.destroyAllWindows()
