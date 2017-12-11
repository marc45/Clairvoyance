from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dropout
from keras import regularizers

import numpy as np
import cv2

#get test data
print ('loading images')
imgW = 200
X = np.zeros((2000, imgW, imgW, 3))
for i in range(X.shape[0]):
    X[i] = cv2.imread('images\\test\\test' + str(i) + '.png') / 255.0

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

wFile = np.load('epoch40w-d.npz')

print (wFile.files)
weights = []
for files in wFile.files:
    weights.append(wFile[files])
    
model.set_weights(weights[0])

model.summary()

model.compile(optimizer='adadelta', loss='binary_crossentropy')

print('training model')
for i in range(0, 200):
    
    model.fit(X, X, initial_epoch=i, epochs=i + 1, batch_size=100)

    print('predicting results')

    encoded_imgs = model.predict(X[53:54])
    img = np.uint8(encoded_imgs[0] * 255)
    cv2.imwrite('images\\result\\epoch' + str(i) + '.png', img)

    if (i % 10 == 0):
        weights = model.get_weights()
        np.savez('epoch' + str(i) + 'w-d', weights)
