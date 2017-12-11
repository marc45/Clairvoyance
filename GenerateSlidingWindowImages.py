import numpy as np
import cv2
import math
import sys

#sys.argv [script name, input video path, output image path, skip X frames, testImgSize, Ksize, scaleFactor]


def RunScript(inpath, outpath, frameSkip, testImgSize, kSize, scaleFactor, imgCount, maxFrames=0, startFrame=0):
    cap = cv2.VideoCapture(inpath)
    imgList = []
    curFrame = 0
    usedFrames = 0

    while(True):
        ret, frame = cap.read()
        if (ret == False):
            break
        
        curFrame += 1
        if (curFrame % frameSkip != 0) or (curFrame < startFrame):
            continue

        usedFrames += 1
        if (maxFrames != 0) and (usedFrames > maxFrames):
            break

        print ('Computing frame ' + str(curFrame) + ' with imgCount ' + str(imgCount) + '...')
        frame = cv2.resize(frame, (int(frame.shape[1] * scaleFactor), int(frame.shape[0] * scaleFactor)))
        
        frame = frame[:400, 900:]

        fX = int(math.floor((frame.shape[0] - testImgSize) / kSize))
        fY = int(math.floor((frame.shape[1] - testImgSize) / kSize))
        for i in range(fX):
            for j in range(fY):
                x1 = i * kSize
                x2 = i * kSize + testImgSize
                y1 = j * kSize
                y2 = j * kSize + testImgSize
                cv2.imwrite(outpath + 'test' + str(imgCount) + '.png', frame[x1:x2, y1:y2])
                imgCount += 1
    
    cap.release()

    print('created ' + str(imgCount) + ' test images.')
    return imgCount

if (len(sys.argv) == 7):
    RunScript(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), 
        float(sys.argv[6]), 0)
    
if (len(sys.argv) == 9):
    RunScript(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), 
        float(sys.argv[6]), 0, maxFrames=int(sys.argv[7]), startFrame=int(sys.argv[8]))

RunScript('videos\\rawfeed.webm', 'images\\test\\', 6, 160, 40, 1, 0, maxFrames=10, startFrame=530)