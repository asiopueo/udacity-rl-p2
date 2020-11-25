import numpy as np

import os
os.environ['KMP_WARNINGS'] = 'FALSE'
import tensorflow as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, BatchNormalization
from keras.layers import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
#from tf.keras.utils import np_utils, to_categorical # ???
import tensorflow.keras.losses as losses


def network_simple():
    model = Sequential()
    # The input layer consists of 37 neurons (35 rays + 2 velocity)
    # The angles are emanated as follows:
    # [,,,,,,]

    model.add( Dense(37, input_shape=(37,) ) )
    model.add( Activation('relu') )
    model.add( Dropout(0.2) )
    model.add( Dense(64) )
    model.add( Activation('relu') )
    model.add( Dropout(0.2) )
    model.add( Dense(64) )
    model.add( Activation('relu') )
    model.add( Dropout(0.2) )
    model.add( Dense(4) )
    model.add( Activation('softmax') )

    model.compile(loss=losses.mean_squared_error, optimizer='sgd')
    return model




"""
def network_pixel():
    model = Sequential()

    model.add( Convolution2D(32, (3,3), input_shape=(), padding='valid', strides=(1,1), activation='relu', data_format='channels_first' )
    model.add( BatchNormalization() )
    model.add( MaxPooling2D( pool_size=(2,2), data_format='channels_first' ) )
    model.add( Convolution2D(64, (3,3), input_shape=(), padding='valid', strides=(1,1), activation='relu', data_format='channels_first' )
    model.add( BatchNormalization() )
    model.add( MaxPooling2D( pool_size=(2,2), data_format='channels_first') )

    model.add( Flatten() )

    model.add( Dense() )
    model.add( Dense() )
    model.add( Dense(4) )

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
"""
