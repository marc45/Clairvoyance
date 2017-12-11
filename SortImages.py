from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Activation
from keras import regularizers

import numpy as np
import cv2
import os

#get test data
imgW = 160
imagesPerBatch = 30

wFile = np.load('epoch50w-d.npz')
weights = []
for files in wFile.files:
    weights.append(wFile[files])


print ('building model')

model = Sequential()
model.add(Conv2D(24, (5, 5), strides=2, padding='same', input_shape=(imgW, imgW, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(32, (5, 5), strides=2, padding='same', activation='relu'))
#model.add(Conv2D(12, (5, 5), strides=2, padding='same', activation='relu'))
model.add(Dropout(0.5))
#model.add(Conv2DTranspose(12, (5, 5), strides=2, padding='same', activation='relu'))
model.add(Conv2DTranspose(32, (5, 5), strides=2, padding='same', activation='relu'))
model.add(Conv2DTranspose(24, (5, 5), strides=2, padding='same', activation='relu'))

model.add(Conv2D(3, (3, 3), padding='same', activation='sigmoid'))

model.set_weights(weights[0])
model.pop()
model.pop()
model.pop()
model.add(Activation('sigmoid'))

model.compile(optimizer='adadelta', loss='binary_crossentropy')

imgList=[]
imgList.append('images\\searchset\\sion.png')
imgList.append('images\\searchset\\sion2.png')
imgList.append('images\\searchset\\sion3.png')
imgList.append('images\\searchset\\sejuani.png')
imgList.append('images\\searchset\\sejuani2.png')
imgList.append('images\\searchset\\thresh.png')
imgList.append('images\\searchset\\thresh2.png')
imgList.append('images\\searchset\\tower.png')
imgList.append('images\\searchset\\xayah.png')
imgList.append('images\\searchset\\xayah2.png')
imgList.append('images\\searchset\\xayah3.png')
imgList.append('images\\searchset\\scuttle.png')

y = np.zeros((len(imgList), imgW, imgW, 3))

for i in range(len(imgList)):
    y[i] = cv2.resize(cv2.imread(imgList[i]), (imgW, imgW)) / 255.0

print('predicting Seach Target')
searchY = model.predict(y)

for batch in range(56):
    print ('loading images in batch ' + str(batch) + ' (' + str(batch * 5000) + '-' + str((batch + 1) * 5000) + ')')
    X = np.zeros((5000, imgW, imgW, 3))
    for i in range(X.shape[0]):
        X[i] = cv2.imread('C:\\Users\\andre\\Desktop\\Code\\League Machine Learning\\images\\test\\test' + str(i + (batch * 5000)) + '.png') / 255.0

    print('predicting Search Results')
    searchX = model.predict(X)

    if (not os.path.exists('images\\searchresult')):
        os.makedirs('images\\searchresult')

    for i in range(y.shape[0]):
        #mean squared error
        rX = np.zeros(searchX.shape[0])
        for s in range(searchX.shape[0]):
            rX[s] = np.sum(np.power(searchX[s] - searchY[i], 2)) / (2 * imgW * imgW * 3)

        sX = np.argsort(rX)
            
        if (not os.path.exists('images\\searchresult\\target' + str(i))):
            os.makedirs('images\\searchresult\\target' + str(i))

        print('saving ' + str(imagesPerBatch) + ' closest matches for ' + imgList[i])
        for s in range(imagesPerBatch):
            img = np.uint8(X[sX[s]] * 255)
            imCount = (imagesPerBatch * batch) + s
            cv2.imwrite('images\\searchresult\\target' + str(i) +'\\img' + str(imCount) +'.png', img)
