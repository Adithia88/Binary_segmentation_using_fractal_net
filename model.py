import os
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler
from keras.layers import MaxPooling2D, UpSampling2D, Convolution2D, Input, merge, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from skimage.io import imsave
from keras import backend as keras
from keras.optimizers import *


def merge(inputs, mode, concat_axis=-1):
    return concatenate(inputs, concat_axis)

def dice_coef(y_true, y_pred, smooth, thresh):
    # y_pred =K.cast((K.greater(y_pred,thresh)), dtype='float32')
    # y_pred = y_pred[y_pred > thresh]=1.0
    y_true_f = y_true  # K.flatten(y_true)
    y_pred_f = y_pred  # K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=(0, 1, 2))
    denom = K.sum(y_true_f, axis=(0, 1, 2)) + K.sum(y_pred_f, axis=(0, 1, 2))
    return K.mean((2. * intersection + smooth) / (denom + smooth))


def dice_loss(smooth, thresh):
    def dice(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred, smooth, thresh)

    return dice


def get_fractalunet(pretrained_weights=None, input_size=(256, 256, 1),f=16):
    inputs = Input(input_size)

    conv1 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv1)

    down1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BatchNormalization()(down1)
    conv2 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2)

    down2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BatchNormalization()(down2)
    conv3 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3)

    down3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BatchNormalization()(down3)
    conv4 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4)

    down4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = BatchNormalization()(down4)
    conv5 = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5)

    up1 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)

    conv6 = BatchNormalization()(up1)
    conv6 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6)

    up2 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)

    conv7 = BatchNormalization()(up2)
    conv7 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7)

    up3 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)

    conv8 = BatchNormalization()(up3)
    conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)

    up4 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)

    conv9 = BatchNormalization()(up4)
    conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)

    # --- end first u block

    down1b = MaxPooling2D(pool_size=(2, 2))(conv9)
    down1b = merge([down1b, conv8], mode='concat', concat_axis=3)

    conv2b = BatchNormalization()(down1b)
    conv2b = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2b)
    conv2b = BatchNormalization()(conv2b)
    conv2b = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2b)

    down2b = MaxPooling2D(pool_size=(2, 2))(conv2b)
    down2b = merge([down2b, conv7], mode='concat', concat_axis=3)

    conv3b = BatchNormalization()(down2b)
    conv3b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3b)
    conv3b = BatchNormalization()(conv3b)
    conv3b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3b)

    down3b = MaxPooling2D(pool_size=(2, 2))(conv3b)
    down3b = merge([down3b, conv6], mode='concat', concat_axis=3)

    conv4b = BatchNormalization()(down3b)
    conv4b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4b)
    conv4b = BatchNormalization()(conv4b)
    conv4b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4b)

    down4b = MaxPooling2D(pool_size=(2, 2))(conv4b)
    down4b = merge([down4b, conv5], mode='concat', concat_axis=3)

    conv5b = BatchNormalization()(down4b)
    conv5b = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5b)
    conv5b = BatchNormalization()(conv5b)
    conv5b = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5b)

    up1b = merge([UpSampling2D(size=(2, 2))(conv5b), conv4b], mode='concat', concat_axis=3)

    conv6b = BatchNormalization()(up1b)
    conv6b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6b)
    conv6b = BatchNormalization()(conv6b)
    conv6b = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6b)

    up2b = merge([UpSampling2D(size=(2, 2))(conv6b), conv3b], mode='concat', concat_axis=3)

    conv7b = BatchNormalization()(up2b)
    conv7b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7b)
    conv7b = BatchNormalization()(conv7b)
    conv7b = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7b)

    up3b = merge([UpSampling2D(size=(2, 2))(conv7b), conv2b], mode='concat', concat_axis=3)

    conv8b = BatchNormalization()(up3b)
    conv8b = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8b)
    conv8b = BatchNormalization()(conv8b)
    conv8b = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8b)

    up4b = merge([UpSampling2D(size=(2, 2))(conv8b), conv9], mode='concat', concat_axis=3)

    conv9b = BatchNormalization()(up4b)
    conv9b = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9b)
    conv9b = BatchNormalization()(conv9b)
    conv9b = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9b)
    conv9b = BatchNormalization()(conv9b)

    outputs = Convolution2D(1, 1, 1, activation='hard_sigmoid', border_mode='same')(conv9b)

    model = Model(inputs=inputs, outputs=outputs)
    
    model_dice = dice_loss(smooth=1e-5, thresh=0.5)
    #model.compile(optimizer = Adam(lr = 1e-5), loss = model_dice, metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-5), loss = "binary_crossentropy", metrics = ['accuracy'])
    
    #model.summary()


    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

