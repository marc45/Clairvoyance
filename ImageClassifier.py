from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Flatten, Dense
from keras import regularizers

import numpy as np
import cv2
import random
from os import listdir

def BuildModel(weightFile, imgW):
    print ('building model')

    model = Sequential()
    model.add(Conv2D(16, (4, 4), strides=2, padding='same', input_shape=(imgW, imgW, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(16, (4, 4), strides=2, padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(16, (4, 4), strides=2, padding='same', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    if(weightFile != ''):
        wFile = np.load(weightFile)
        weights = []
        for files in wFile.files:
            weights.append(wFile[files])
        model.set_weights(weights[0])

    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model



# def PredictClassifier(weightFile, predictLocation):
#     imgW = 64
#     directoryList = []
#     directoryList.append('images\\leagueset\\sion\\lumberjack')
#     directoryList.append('images\\leagueset\\sejuani\\poro')
#     directoryList.append('images\\leagueset\\thresh\\base')
#     directoryList.append('images\\leagueset\\xayah\celestial')
#     X = np.zeros((0, imgW, imgW, 3))
#     Y = np.zeros((0, len(directoryList)))

#     model = BuildModel(weightFile, imgW)

#     rX = model.predict(X)

#     pX = np.argmax(rX, axis=1)
#     pY = np.argmax(Y, axis=1)
#     n = 0
#     for i in range(pY.shape[0]):
#         if(np.sum(pY[i]) == 0) and (rX[i, pX[i]] < 0.5):
#             n += 1
#         elif(pX[i] == pY[i]) and (rX[i, pX[i]] >= 0.5):
#             n += 1
#         #else:
#             #print([i, Y[i,:]])

#     print('training set accuracy: ' +str(n) + '/' + str(Y.shape[0]))

#     files = [f for f in listdir(predictLocation) if f.endswith('.png')]
#     tX = np.zeros((len(files), imgW, imgW, 3))
#     for i in range(len(files)):
#         tX[i] = cv2.resize(cv2.imread(predictLocation + '\\' + files[i]), (imgW, imgW)) / 255.0

#     print('loaded ' + str(tX.shape[0]) + ' test sets')

#     rX = model.predict(tX)

#     np.savetxt('test2.out', rX)
#     pX = 1 * (np.amax(rX, axis=1) >= 0.5)

#     print('test set accuracy: ' +str(np.sum(pX)) + '/' + str(pX.shape[0]))

#     for i in range(len(files)):
#         if (pX[i] == 1):
#             img = cv2.imread('images\\searchresult\\target8\\' + files[i])
#             cv2.imwrite('images\\result\\search' + str(i) + '.png', img)

def BatchTrainClassifier(batchList, negativeDir, extraTrainingCount, endEpoch):
    # batchList = [[training data folder, weight file name], ...]
    imgW = 64
    filesPerBatch = int(extraTrainingCount / (len(batchList) - 1))
    
    negX = 0
    negY = 0
    # we only need to load the negative examples once
    files = [f for f in listdir(negativeDir) if f.endswith('.png')]
    negX = np.zeros((len(files), imgW, imgW, 3))
    negY = np.zeros((len(files), 1))
    for i in range(len(files)):
        negX[i] = cv2.resize(cv2.imread(negativeDir + '\\' + files[i]), (imgW, imgW)) / 255.0

    for b in range(len(batchList)):
        if (b != 2):
            continue
        # load positive examples for this batch
        files = [f for f in listdir(batchList[b][0]) if f.endswith('.png')]
        X = np.zeros((len(files), imgW, imgW, 3))
        Y = np.ones((len(files), 1))
        for i in range(len(files)):
            X[i] = cv2.resize(cv2.imread(batchList[b][0] + '\\' + files[i]), (imgW, imgW)) / 255.0
        
        X = np.append(X, negX, axis=0)
        Y = np.append(Y, negY, axis=0)

        # grab extraTrainingCount / batch list examples from other classifiers
        for j in range(len(batchList)):
            if (b != j):
                files = [f for f in listdir(batchList[j][0]) if f.endswith('.png')]
                A = [i for i in range(len(files))]
                random.shuffle(A)
                bX = np.zeros((filesPerBatch, imgW, imgW, 3))
                bY = np.zeros((filesPerBatch, 1))
                for i in range(min(len(files), filesPerBatch)):
                    bX[i] = cv2.resize(cv2.imread(batchList[j][0] + '\\' + files[A[i]]), (imgW, imgW)) / 255.0
                X = np.append(X, bX, axis=0)
                Y = np.append(Y, bY, axis=0)
                

        print('training model with set ' + batchList[b][0])
        model = BuildModel('', imgW)
        model.fit(X, Y, epochs=endEpoch, batch_size=500)

        print('saving weights to ' + batchList[b][1] + '.npz')
        weights = model.get_weights()
        np.savez(batchList[b][1], weights)
                

def TrainClassifier(weightFile, trainDir, negativeDir, endEpoch):
    imgW = 64

    model = BuildModel('', imgW)

    print('loading test data')
    files = [f for f in listdir(trainDir) if f.endswith('.png')]
    X = np.zeros((len(files), imgW, imgW, 3))
    Y = np.zeros((len(files), 1))
    for i in range(len(files)):
        X[i] = cv2.resize(cv2.imread(trainDir + '\\' + files[i]), (imgW, imgW)) / 255.0
        Y[i, 0] = 1

    for dirs in negativeDir:
        files = [f for f in listdir(dirs) if f.endswith('.png')]
        tX = np.zeros((len(files), imgW, imgW, 3))
        tY = np.zeros((len(files), 1))
        for i in range(len(files)):
            tX[i] = cv2.resize(cv2.imread(dirs + '\\' + files[i]), (imgW, imgW)) / 255.0
        X = np.append(X, tX, axis=0)
        Y = np.append(Y, tY, axis=0)
    print('loaded ' + str(X.shape[0]) + ' training sets')

    print('training model')
    for i in range(endEpoch):
        
        model.fit(X, Y, initial_epoch=i, epochs=i + 1, batch_size=500)

        if (i % 10 == 9):
            print('saving weights to ' + weightFile + '.npz')
            weights = model.get_weights()
            np.savez(weightFile, weights)

