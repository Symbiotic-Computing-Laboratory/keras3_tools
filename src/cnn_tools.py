'''
Andrew H. Fagg

CNN-based classifiers

'''

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import random

import keras
from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, BatchNormalization
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Concatenate
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, SpatialDropout2D
from keras.layers import Convolution3D, MaxPooling3D, AveragePooling3D, GlobalMaxPooling3D, SpatialDropout3D

conv_operators = {'C1': (Convolution1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D),
                  'C2': (Convolution2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, SpatialDropout2D),
                  'C3': (Convolution3D, MaxPooling3D, AveragePooling3D, GlobalMaxPooling3D, SpatialDropout3D),
                  }

#import re

#import sklearn.metrics

from fully_connected_tools import *

def create_cnn_stack(tensor,
                     conv_filters:[int],
                     conv_kernel_size:[int],
                     conv_pool:[int],
                     conv_strides:[int],
                     conv_type:str='C2',
                     name_base:str='',
                     conv_activation:str='elu',
                     regularizer=None,
                     s_dropout=None,
                     padding='valid',
                     batch_normalization=False,
                     global_max_pool=False):
    '''
    Create a linear convolutional stack

    '''
    conv, mp, ap, gmp, sd = conv_operators[conv_type]
    
    # Loop over all convolutional layers
    for i, (f,k,p,s) in enumerate(zip(conv_filters, conv_kernel_size, conv_pool, conv_strides)):
        tensor = conv(filters=f,
                      kernel_size=k,
                      strides=1,
                      padding=padding,
                      use_bias=True,
                      kernel_initializer='truncated_normal',
                      bias_initializer='zeros',
                      name='%sC%d'%(name_base,i),
                      activation=conv_activation,
                      kernel_regularizer=regularizer)(tensor)
        
        # Possible dropout
        if s_dropout is not None:
            tensor = sd(s_dropout)(tensor)

        if batch_normalization:
            tensor = bn()(tensor)

        if p is not None and p > 1:
            tensor = mp(p,
                        strides=s,
                        padding=padding)(tensor)
            
    if global_max_pool:
        tensor=gmp()(tensor)
    return tensor
