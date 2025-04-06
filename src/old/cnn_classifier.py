'''
Andrew H. Fagg

CNN-based classifiers

OLD MATERIAL
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
from keras.layers import InputLayer, Input, BatchNormalization, SpatialDropout2D
from keras.layers import Convolution2D, Dense, MaxPooling2D, AveragePooling2D, Flatten, BatchNormalization, Dropout, SpatialDropout2D, Concatenate, GlobalMaxPooling2D

import re

import sklearn.metrics

from deep_networks import *

def create_cnn_network(image_size, nchannels, 
                       name_base='',
                       conv_layers=[],                                  
                       conv_activation='elu',
                       dense_layers=[],
                       dense_activation='elu',
                       lrate=.0001,
                       lambda_l2=None,
                       p_dropout=None,
                       p_spatial_dropout=None,
                       padding='valid',
                       flatten=True):
    model = Sequential()
    model.add(InputLayer(shape=(image_size[0], image_size[1], nchannels),
                        name='%sinput'%(name_base)))
    
    if lambda_l2 is not None:
        lambda_l2 = tf.keras.regularizers.l2(lambda_l2)
    
    # Loop over all convolutional layers
    i = 0
    for l in conv_layers:
        print("Layer", i, l)

        model.add(Convolution2D(filters=l['filters'],
                                kernel_size=l['kernel_size'],
                                strides=(1,1),
                                padding=padding,
                                use_bias=True,
                                kernel_initializer='truncated_normal',
                                bias_initializer='zeros',
                                name='%sC%d'%(name_base,i),
                                activation=conv_activation,
                                kernel_regularizer=lambda_l2))
        
        if p_spatial_dropout is not None:
            model.add(SpatialDropout2D(p_spatial_dropout))
                      
        
        if 'batch_normalization' in l and l['batch_normalization']:
            model.add(BatchNormalization())

        if l['pool_size'] is not None and l['pool_size'][0] > 1:
            model.add(MaxPooling2D(pool_size=l['pool_size'],
                               strides=l['strides'],
                               padding=padding))
          
        i=i+1
        
    # Flatten
    if flatten:
        model.add(Flatten())
    else:
        model.add(GlobalMaxPooling2D())
    
    # Loop over dense layers
    i = 0
    for l in dense_layers:
        model.add(Dense(units=l['units'],
                        activation=dense_activation,
                        use_bias=True,
                        kernel_initializer='truncated_normal',
                        bias_initializer='zeros',
                        name='%sD%d'%(name_base,i),
                        kernel_regularizer=lambda_l2))

        if 'batch_normalization' in l and l['batch_normalization']:
            model.add(BatchNormalization())
        
        if p_dropout is not None:
            model.add(Dropout(p_dropout,
                              name='%sDR%d'%(name_base,i)))
            
        i=i+1
        
    return model

def create_cnn_stack(tensor, conv_layers,
                     name_base='',
                     regularizer=None,
                     s_dropout=None,
                     conv_activation='elu',
                     padding='valid'):
    '''
    Create a linear convolutional stack

    TODO: call from above function
    '''
    
    # Loop over all convolutional layers
    for i, l in enumerate(conv_layers):
        print("Layer", i, l)
        tensor = Convolution2D(filters=l['filters'],
                                kernel_size=l['kernel_size'],
                                strides=(1,1),
                                padding=padding,
                                use_bias=True,
                                kernel_initializer='truncated_normal',
                                bias_initializer='zeros',
                                name='%sC%d'%(name_base,i),
                                activation=conv_activation,
                                kernel_regularizer=regularizer)(tensor)
        
        # Possible dropout
        if s_dropout is not None:
            tensor = SpatialDropout2D(s_dropout)(tensor)

        if 'batch_normalization' in l and l['batch_normalization']:
            tensor = BatchNormalization()(tensor)

        if l['pool_size'] is not None and l['pool_size'][0] > 1:
            tensor = MaxPooling2D(pool_size=l['pool_size'],
                               strides=l['strides'],
                               padding=padding)(tensor)
    return tensor
        

def create_cnn_classifier_network(image_size, nchannels, 
                                  name_base='',
                                  conv_layers=[],                                  
                                  conv_activation='elu',
                                  dense_layers=[],
                                  dense_activation='elu',
                                  n_classes=2, 
                                  lrate=.0001,
                                  lambda_l2=None,
                                  p_dropout=None,
                                  p_spatial_dropout=None,
                                  output_activation='softmax',
                                  loss='categorical_crossentropy',
                                  metrics=['categorical_accuracy'],
                                  padding='valid',
                                  flatten=True):
    
    # Create base model
    model = create_cnn_network(image_size, nchannels,
                              name_base,
                              conv_layers,
                              conv_activation,
                              dense_layers,
                              dense_activation,
                              lrate,
                              lambda_l2,
                              p_dropout,
                               p_spatial_dropout,
                               padding=padding,
                               flatten=flatten)
    # Output layer
    model.add(Dense(units=n_classes,
                    activation=output_activation,
                    use_bias=True,
                    bias_initializer='zeros',
                    name='%soutput'%(name_base)))
    
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, #beta_1=.9, beta_2=0.999,
                                   #epsilon=None, decay=0.0,
                                   amsgrad=False)
    
    # Asssuming right now two outputs
    model.compile(loss=loss, optimizer=opt,
                 metrics=metrics) 
    return model

