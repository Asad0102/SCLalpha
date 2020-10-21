import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import keras
import time
import sys
import math
import scipy
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import time



def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def reshape_3Dto4D(A):
    return A.reshape(1, A.shape[0] , A.shape[1], A.shape[2])

wav_pointer = 10
model_dir = '/Users/michailisaakidis/Desktop/SoundLocalization/models/50e_model.h5'
sound_dir = '/Users/michailisaakidis/Desktop/SoundLocalization/convertedSounds/my_data.wav.npy'


model = keras.models.load_model(model_dir)
input = model.input
output = model.output

soundbatch = np.load(sound_dir)


print(soundbatch.shape)


#print(soundbatch[0][4].min())
#print(soundbatch[0][4].max())
#print(np.mean(soundbatch[0][4]))
soundbatch = reshape_3Dto4D(soundbatch)

#startTime = time.time()
pred = model.predict(soundbatch)
#elapsedTime = time.time() - startTime

sed_pred = reshape_3Dto2D(pred[0]) > 0.50
#doa_pred = reshape_3Dto2D(pred[1])

print(sed_pred)
#print("inference time (sec): " + str(elapsedTime))
