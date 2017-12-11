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
    model.add(Conv2D(32, (11, 11), strides=4, padding='same', input_shape=(imgW, imgW, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (5, 5), strides=2, padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), strides=2, padding='same', activation='relu'))
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


def BatchTrainClassifier(batchList, negativeDir, extraTrainingCount, endEpoch):
    # batchList = [[training data folder, weight file name], ...]
    imgW = 96
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

