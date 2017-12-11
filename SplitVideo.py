import numpy as np
import cv2
import sys

###sys.argv[script name, input file, output path, seconds to split] ###

def WriteVideo(imgList, name, vidSize):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(name, fourcc, 30.0, vidSize)
    print(name)
    for i in range(len(imgList)):
        out.write(imgList[i])
    out.release()

def RunScript():
    if (len(sys.argv) != 4):
        print(sys.argv)
        return

    numSplits = 0
    splitframes = int(sys.argv[3]) * 30
    cap = cv2.VideoCapture(sys.argv[1])
    imgList = []
    vidSize = (0, 0)
    frameSkip = 10
    count = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if (ret == False):
            break

        count += 1
        if(count % frameSkip != 0):
            continue
        imgList.append(frame)

        if (len(imgList) >= splitframes):
            print('Writing split ' + str(numSplits))
            vidSize = (frame.shape[1], frame.shape[0])
            WriteVideo(imgList, sys.argv[2] + str(numSplits) + '.avi', vidSize)
            numSplits += 1
            imgList = []
    
    if (len(imgList) > 0):
        WriteVideo(imgList, sys.argv[2] + str(numSplits) + '.avi', vidSize)

    cap.release()
    print('Split Video into ' + str(numSplits) + ' pieces.')

RunScript()