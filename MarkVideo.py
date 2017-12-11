from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Flatten, Dense
from keras import regularizers

import ImageClassifier
import numpy as np
import cv2
import math
from time import time
from os import listdir, path, makedirs

colorList = []
colorList.append([255, 0, 0])
colorList.append([0, 255, 0])
colorList.append([0, 0, 255])
colorList.append([127, 127, 0])
colorList.append([127, 0, 127])
colorList.append([0, 127, 127])
colorList.append([127, 63, 63])
colorList.append([63, 127, 63])
colorList.append([63, 63, 127])
colorList.append([255, 255, 0])
colorList.append([255, 0, 255])
colorList.append([0, 255, 255])
colorList.append([127, 0, 0])
colorList.append([0, 127, 0])
colorList.append([0, 0, 127])

def FindBox( npMat ):
    cor1 = [0, 0]
    cor2 = [int(npMat.shape[0]), int(npMat.shape[1])]

    start = np.unravel_index(np.argmax(npMat), npMat.shape)
    for i in range(start[0], 0, -1):
        cor1[0] = i
        if (npMat[cor1[0], start[1]] < npMat[start[0], start[1]]):
            break
    for i in range(start[1], 0, -1):
        cor1[1] = i
        if (npMat[start[0], cor1[1]] < npMat[start[0], start[1]]):
            break
    for i in range(start[0], npMat.shape[0]):
        cor2[0] = i
        if (npMat[cor2[0], start[1]] < npMat[start[0], start[1]]):
            break
    for i in range(start[1], npMat.shape[1]):
        cor2[1] = i
        if (npMat[start[0], cor2[1]] < npMat[start[0], start[1]]):
            break
    return (int((cor1[1] + cor2[1]) / 2), int((cor1[0] + cor2[0]) / 2))

def RenderOverlay(frame, boxList, models, boxRadius):

    for m in range(len(models)):
        #We only show the largest concentration of success per model
        if (np.sum(boxList[m]) > 0):
            center = FindBox(boxList[m])
            a = (int(max(0, min(frame.shape[1], center[0] * 10 - boxRadius))), int(max(0, min(frame.shape[0], center[1] * 10 - boxRadius))))
            b = (int(max(0, min(frame.shape[1], center[0] * 10 + boxRadius))), int(max(0, min(frame.shape[0], center[1] * 10 + boxRadius))))
            cv2.rectangle(frame, a, b, colorList[m], 2)

        #write a legend in the top left corner
        cv2.rectangle(frame, (4, 4 + (m * 25)), (16, 16 + (m * 25)), colorList[m], -1)
        cv2.putText(frame, models[m][1], (20, 16 + (m * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

def MarkVideo(inFile, outFile, models, startFrame=0, frameSkip=1, numFrames=1, windowSize=160, step=40, threshold=0.8):
    ### models = [[weightFile, name], ...]
    
    imgW = 64

    convNetList = []
    for model in models:
        convNetList.append(ImageClassifier.BuildModel(model[0], imgW))

    curFrame = 0
    usedFrames = 0
    windowHalfSize = windowSize / 2
    imgCount = 0
    vidCount = 0

    cap = cv2.VideoCapture(inFile)
    out = False
    initOutFile = False
    boxList = False
    initBoxList = False

    timeStart = time()

    while(True):
        ret, frame = cap.read()

        curFrame += 1

        if (initBoxList == False):
            initBoxList = True
            boxList = np.zeros((len(convNetList),int(frame.shape[0] / 10),int(frame.shape[1] / 10)))
        if (initOutFile == False):
            initOutFile = True
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(outFile, fourcc, 30.0, (frame.shape[1], frame.shape[0]))

        if (ret == False) or (usedFrames >= numFrames and numFrames != -1):
            break
        if (curFrame < startFrame):
            continue
        if (curFrame % frameSkip != frameSkip - 1):
            RenderOverlay(frame, boxList, models, windowHalfSize)
            out.write(frame)
            continue
        print('processing frame ' + str(curFrame))
        usedFrames += 1

        #Split frame into slices to pass into our models
        fX = int(math.floor((frame.shape[0] - windowSize) / step))
        fY = int(math.floor((frame.shape[1] - windowSize) / step))
        imgList = np.zeros((fX * fY, imgW, imgW, 3))
        for i in range(fX):
            for j in range(fY):
                x1 = i * step
                x2 = i * step + windowSize
                y1 = j * step
                y2 = j * step + windowSize
                imgList[(i * fY) + j] = cv2.resize(frame[x1:x2, y1:y2], (imgW, imgW))
        imgList = imgList / 255.0    

        #process slices and mark results on the frame
        for m in range(len(models)):
            boxList[m] = np.zeros((int(frame.shape[0] / 10),int(frame.shape[1] / 10)))
            
            rX = convNetList[m].predict(imgList)

            for i in range(fX):
                for j in range(fY):
                    t = (i * fY) + j
                    if (rX[t, 0] >= threshold):
                        x1 = int((i * step) / 10)
                        x2 = int((i * step + windowSize) / 10)
                        y1 = int((j * step) / 10)
                        y2 = int((j * step + windowSize) / 10)
                        boxList[m, x1:x2, y1:y2] += 1

        RenderOverlay(frame, boxList, models, windowHalfSize)
        out.write(frame)
            
    out.release()
    print('video took ~{0:.2f}s'.format(time() - timeStart))
    
def GetOutputsFromVideo(inFile, outDir, models, startFrame=0, frameSkip=1, numFrames=1, windowSize=160, step=40, threshold=0.8):

    imgW = 64
    convNetList = []
    for model in models:
        convNetList.append(ImageClassifier.BuildModel(model[0], imgW))
        if (not path.exists(outDir + '\\' + model[1])):
            makedirs(outDir + '\\' + model[1])

    frameList = []
    curFrame = 0
    usedFrames = 0
    imgCount = 0

    cap = cv2.VideoCapture(inFile)

    while(True):
        ret, frame = cap.read()
        if (ret == False) or (usedFrames >= numFrames):
            break

        curFrame += 1
        if (curFrame % frameSkip != 0) or (curFrame < startFrame):
            continue
        print('processing frame ' + str(curFrame))
        usedFrames += 1

        #Split frame into slices to pass into our models
        fX = int(math.floor((frame.shape[0] - windowSize) / step))
        fY = int(math.floor((frame.shape[1] - windowSize) / step))
        imgList = np.zeros((fX * fY, imgW, imgW, 3))
        for i in range(fX):
            for j in range(fY):
                x1 = i * step
                x2 = i * step + windowSize
                y1 = j * step
                y2 = j * step + windowSize
                imgList[(i * fY) + j] = cv2.resize(frame[x1:x2, y1:y2], (imgW, imgW))
        imgList = imgList / 255.0 

        for m in range(len(models)):
            rX = convNetList[m].predict(imgList)

            for i in range(fX):
                for j in range(fY):
                    t = (i * fY) + j
                    if (rX[t, 0] >= threshold):
                        x1 = i * step
                        x2 = i * step + windowSize
                        y1 = j * step
                        y2 = j * step + windowSize
                        cv2.imwrite(outDir + '\\' + models[m][1] + '\\res' + str(imgCount) + '.png', frame[x1:x2, y1:y2])
                        imgCount += 1
