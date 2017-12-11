from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Flatten, Dense
from keras import regularizers

import numpy as np
import cv2
from os import listdir
from time import time
import ImageClassifier

batchList = []
batchList.append(['images\\leagueset\\sion\\lumberjack', 'weights\\weight-sion'])
batchList.append(['images\\leagueset\\sejuani\\poro', 'weights\\weight-sejuani'])
batchList.append(['images\\leagueset\\thresh\\base', 'weights\\weight-thresh'])
batchList.append(['images\\leagueset\\xayah\\celestial', 'weights\\weight-xayah'])
batchList.append(['images\\leagueset\\jinx\\base', 'weights\\weight-jinx'])
batchList.append(['images\\leagueset\\leesin\\base', 'weights\\weight-leesin'])
batchList.append(['images\\leagueset\\ezreal\\base', 'weights\\weight-ezreal'])
batchList.append(['images\\leagueset\\bard\\base', 'weights\\weight-bard'])
batchList.append(['images\\leagueset\\minions\\redcaster', 'weights\\weight-MRedCaster'])
batchList.append(['images\\leagueset\\minions\\redcannon', 'weights\\weight-MRedCannon'])

timeStart = time()
ImageClassifier.BatchTrainClassifier(batchList, 'images\\none', 400, 250)
print('training took ~{0:.2f}s'.format(time() - timeStart))