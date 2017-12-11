import numpy as np
import cv2
import sys


cap = cv2.VideoCapture(sys.argv[1])

ret, frame = cap.read()

print(frame.shape)

cap.release()
