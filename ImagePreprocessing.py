import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import math

def CreateTestSetFromImage( filename, windowSize, step, testImgSize, grayscale ):
    img = cv2.imread(filename)
    
    if (grayscale):
        imgLen = windowSize[0] * windowSize[1]
        testImgLen = testImgSize[0] * testImgSize[1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        imgLen = windowSize[0] * windowSize[1] * 3
        testImgLen = testImgSize[0] * testImgSize[1] * 3
    
    stepsX = math.floor((img.shape[0] - windowSize[0]) / step)
    stepsY = math.floor((img.shape[1] - windowSize[1]) / step)

    X = np.zeros((stepsX * stepsY, testImgLen))

    for i in range(stepsY):
        for s in range(stepsX):
            tmp = img[(s * step):(s * step) + windowSize[0],(i * step):(i * step) + windowSize[1]]
            X[(i * stepsX) + s, :] = cv2.resize(tmp, testImgSize).reshape(1, testImgLen)
    
    return X

def MarkImageWithResults( filename, windowSize, step, rX ):
    img = cv2.imread(filename)

    stepsX = math.floor((img.shape[0] - windowSize[0]) / step)
    stepsY = math.floor((img.shape[1] - windowSize[1]) / step)

    for i in range(rX.shape[0]):
        #if (np.sum(rX[i, :]) != 0):
        if (rX[i, 0] == 1):
            t = math.floor(i / stepsX)
            s = i % stepsX
            tmp = img[(s * step):(s * step) + windowSize[0],(t * step):(t * step) + windowSize[1]]
            tmp[0, :, 0] = 255
            tmp[-1, :, 0] = 255
            tmp[:, 0, 0] = 255
            tmp[:, -1, 0] = 255
            img[(s * step):(s * step) + windowSize[0],(t * step):(t * step) + windowSize[1]] = tmp
    
    return img
    

def ConvertImageToKMeansSet( img, scaleFactor ):
    img = cv2.resize(img, (int(img.shape[1] * scaleFactor), int(img.shape[0] * scaleFactor)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 150, 200)

    X = np.zeros((0, 2))
    n = np.zeros((1000, 2))
    j = 0

    for i in range(img.shape[0]):
        for s in range(img.shape[1]):
            if (img[i, s] != 0):
                n[j, 0] = i
                n[j, 1] = s
                j += 1
                if (j == 1000):
                    X = np.append(X, n, axis=0)
                    j = 0
    
    X = np.append(X, n[0:(1000 - j), :], axis=0)

    return X

def MarkImageFromCentroids( filename, K, pX, imgSize):
    img = cv2.imread(filename)

    for i in range(len(K)):
        if (pX[i, 0] == 1):
            x1 = int(max(0, K[i][0] - (imgSize[0] / 2)))
            y1 = int(max(0, K[i][1] - (imgSize[1] / 2)))
            x2 = int(min(img.shape[0], K[i][0] + (imgSize[0] / 2)))
            y2 = int(min(img.shape[1], K[i][1] + (imgSize[1] / 2)))
            img[x1, y1:y2, 0] = 255
            img[x2 - 1, y1:y2, 0] = 255
            img[x1:x2, y1, 0] = 255
            img[x1:x2, y2 - 1, 0] = 255

    return img
    
def GetImagesFromCentroids( filename, K, imgSize, setSize ):
    img = cv2.imread(filename)

    imgLen = setSize[0] * setSize[1] * 3

    X = np.zeros((K.shape[0], imgLen))
    t = 64

    stitch = np.zeros((640,640,3), np.uint8)

    for i in range(len(K)):
        kX = min(img.shape[0] - (imgSize[0] / 2), max((imgSize[0] / 2), K[i][0]))
        kY = min(img.shape[1] - (imgSize[1] / 2), max((imgSize[1] / 2), K[i][1]))
        x1 = int(max(0, kX - (imgSize[0] / 2)))
        y1 = int(max(0, kY - (imgSize[1] / 2)))
        x2 = int(min(img.shape[0], kX + (imgSize[0] / 2)))
        y2 = int(min(img.shape[1], kY + (imgSize[1] / 2)))
        a = int(i / 10) * t
        b = (i % 10) * t
        stitch[a:a + t, b:b + t] = cv2.resize(img[x1:x2, y1:y2], (t, t))
        X[i, :] = np.reshape(cv2.resize(img[x1:x2, y1:y2], setSize), (1, imgLen))
    cv2.imshow('aa', stitch)
    cv2.waitKey(0)
    return X

def GetRawImagesFromCentroids( img, K, imgSize ):
    imgList = []

    for i in range(len(K)):
        kX = min(img.shape[0] - (imgSize[0] / 2), max((imgSize[0] / 2), K[i][0]))
        kY = min(img.shape[1] - (imgSize[1] / 2), max((imgSize[1] / 2), K[i][1]))
        x1 = int(max(0, kX - (imgSize[0] / 2)))
        y1 = int(max(0, kY - (imgSize[1] / 2)))
        x2 = int(min(img.shape[0], kX + (imgSize[0] / 2)))
        y2 = int(min(img.shape[1], kY + (imgSize[1] / 2)))
        imgList.append(img[x1:x2, y1:y2].copy())

    return imgList

def LoadImageToMatrix( X, Y, y, imgSize, shouldCrop, grayscale, filename ):
    img = cv2.imread(filename)
    if (grayscale):
        imgLen = imgSize[0] * imgSize[1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        imgLen = imgSize[0] * imgSize[1] * 3

    X = np.append(X, cv2.resize(img, imgSize).reshape(1, imgLen), axis=0)
    Y = np.append(Y, y.copy(), axis=0)
    
    #crop image to duplicate results
    if (shouldCrop):
        img2 = img[10:img.shape[0], 0:img.shape[1]]
        img3 = img[0:img.shape[0], 10:img.shape[1]]
        img4 = img[10:img.shape[0], 10:img.shape[1]]
        img5 = img[0:(img.shape[0] - 10), 0:img.shape[1]]
        img6 = img[0:img.shape[0], 0:(img.shape[1] - 10)]
        img7 = img[0:(img.shape[0] - 10), 0:(img.shape[1] - 10)]
        X = np.append(X, cv2.resize(img2, imgSize).reshape(1, imgLen), axis=0)
        Y = np.append(Y, y.copy(), axis=0)
        X = np.append(X, cv2.resize(img3, imgSize).reshape(1, imgLen), axis=0)
        Y = np.append(Y, y.copy(), axis=0)
        X = np.append(X, cv2.resize(img4, imgSize).reshape(1, imgLen), axis=0)
        Y = np.append(Y, y.copy(), axis=0)
        X = np.append(X, cv2.resize(img5, imgSize).reshape(1, imgLen), axis=0)
        Y = np.append(Y, y.copy(), axis=0)
        X = np.append(X, cv2.resize(img6, imgSize).reshape(1, imgLen), axis=0)
        Y = np.append(Y, y.copy(), axis=0)
        X = np.append(X, cv2.resize(img7, imgSize).reshape(1, imgLen), axis=0)
        Y = np.append(Y, y.copy(), axis=0)
    return X, Y
#end def LoadImageToMatrix

def LoadData( fileout, imgSize, createArtificialImg, grayscale, imgList, trainPercent ):
    #fileout - str of base filename, will output 2-4 files with <filename>X/Y/TestX/TestY
    #imgSize - (x, y) size of compressed img
    #createArtificialImg - true/false crops the input image to create more training sets
    #imgList - list containing [base file name, file ext, img count]
    #trainPercent - 0-1, how many images should go to the training or test set, val of 1 means no TestX/TestY file
    if (grayscale):
        imgLen = imgSize[0] * imgSize[1]
    else:
        imgLen = imgSize[0] * imgSize[1] * 3

    X = np.ndarray(shape=(0, imgLen))
    Xtest = np.ndarray(shape=(0, imgLen))
    Y = np.ndarray(shape=(0, len(imgList)))
    Ytest = np.ndarray(shape=(0, len(imgList)))

    for i in range(len(imgList)):
        y = np.zeros((1, len(imgList)))
        y[0, i] = 1
        nums = [j for j in range(imgList[i][2])]
        random.shuffle(nums)
        for s in nums:
            if (s / len(nums) > trainPercent):
                Xtest, Ytest = LoadImageToMatrix(Xtest, Ytest, y, imgSize, createArtificialImg, grayscale, imgList[i][0] + str(s) + imgList[i][1])
            else:
                X, Y = LoadImageToMatrix(X, Y, y, imgSize, createArtificialImg, grayscale, imgList[i][0] + str(s) + imgList[i][1])
    



    np.save(fileout + 'X', X)
    np.save(fileout + 'Y', Y)
    if (trainPercent < 1):
        np.save(fileout + 'TestX', Xtest)
        np.save(fileout + 'TestY', Ytest)