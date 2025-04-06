'''
Fully connected network tools

'''

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import time

import argparse
import pickle

# Keras3
from keras.layers import InputLayer, Dense, Dropout
from keras.models import Sequential


def create_regularizer(lambda1:float=None, lambda2:float=None):
    if lambda1 is None:
        if lambda2 is None:
            return None
        else:
            return keras.regularizers.l2(lambda2)
    else:
        if lambda2 is None:
            return keras.regularizers.l1(lambda1)
        else:
            return keras.regularizers.l1l2(lambda1, lambda2)

def create_dense_stack(tensor,
                       n_hidden,
                       name='D',
                       activation='elu',
                       lambda1:float=None,
                       lambda2:float=None,
                       dropout=None,
                       name_last='output',
                       activation_last=None,
                       batch_normalization=False):

    regularizer = create_regularizer(lambda1, lambda2)

    if activation_last is None:
        activation_last = activation
    
    for i, n in enumerate(n_hidden[:-1]):
        if batch_normalization:
            tensor = BatchNormalization()(tensor)

        tensor = Dense(n, use_bias=True,
                       bias_initializer='zeros',
                       name="%s_%d"%(name,i),
                       activation=activation,
                       kernel_regularizer=regularizer,
                       kernel_initializer='truncated_normal')(tensor)
        if dropout is not None:
            tensor = Dropout(rate=dropout,
                             name="%s_drop_%d"%(name,i))(tensor)
                       

    # Last layer
    if batch_normalization:
        tensor = BatchNormalization()(tensor)

    tensor = Dense(n_hidden[-1], use_bias=True,
                   bias_initializer='zeros',
                   name=name_last,
                   activation=activation_last,
                   kernel_regularizer=regularizer,
                   kernel_initializer='truncated_normal')(tensor)
    
    
    return tensor
