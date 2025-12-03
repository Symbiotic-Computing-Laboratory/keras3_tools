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
from fully_connected_tools import *

class ConvolutionalNeuralNetwork:
    conv_operators = {'C1': (Convolution1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D),
                      'C2': (Convolution2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, SpatialDropout2D),
                      'C3': (Convolution3D, MaxPooling3D, AveragePooling3D, GlobalMaxPooling3D, SpatialDropout3D),
                      }

    @staticmethod
    def create_cnn_stack(tensor,
                         conv_filters:[int],
                         conv_kernel_size:[int],
                         conv_pool_average:[int],
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
        # Access the layer constructors for the dimensionality of the input
        conv, mp, ap, gmp, sd = ConvolutionalNeuralNetwork.conv_operators[conv_type]

        conv_activation = FullyConnectedNetwork.translate_activation_function(conv_activation)


        # Deal with case where conv_strides is None (translate to all 1s)
        if conv_strides is None:
            conv_strides = [1]*min([len(conv_filters), len(conv_kernel_size), len(conv_pool)])

        # Also deal with conv_pool_average being None
        if conv_pool_average is None:
            conv_pool_average = [None]*min([len(conv_filters), len(conv_kernel_size), len(conv_pool)])
    
        # Loop over all convolutional layers
        for i, (f,k,avg_p,p,s) in enumerate(zip(conv_filters, conv_kernel_size, conv_pool_average, conv_pool, conv_strides)):
            
            # Average pooling
            if avg_p is not None and avg_p > 1:
                tensor = ap(avg_p,
                            strides=avg_p,
                            padding=padding)(tensor)

            # Convolution
            tensor = conv(filters=f,
                          kernel_size=k,
                          strides=s,
                          padding=padding,
                          use_bias=True,
                          kernel_initializer='truncated_normal',
                          bias_initializer='zeros',
                          name='%s%d'%(name_base,i),
                          activation=conv_activation,
                          kernel_regularizer=regularizer)(tensor)
        
            # Possible dropout
            if s_dropout is not None:
                tensor = sd(s_dropout)(tensor)

            # Batch Norm
            if batch_normalization:
                tensor = BatchNormalization()(tensor)

            # Max pooling
            if p is not None and p > 1:
                tensor = mp(p,
                            strides=p,
                            padding=padding)(tensor)

        # Tack on global max pooling
        if global_max_pool:
            tensor=gmp()(tensor)
            
        return tensor


    @staticmethod
    def create_cnn_network(input_shape=None,
                           conv_kernel_size=[3,3],
                           conv_padding='valid',
                           conv_number_filters=[8,16],
                           conv_activation='elu',
                           conv_pool_average_size=None,
                           conv_pool_size=[0,2],
                           conv_strides=None,
                           spatial_dropout=None,
                           conv_batch_normalization=False,
                           n_hidden=[10,2],
                           output_shape=[1],
                           dropout_input=False,
                           name_base='',
                           activation='elu',
                           lambda1:float=None,
                           lambda2:float=None,
                           dropout=None,
                           name_last='output',
                           activation_last=None,
                           batch_normalization=False,
                           learning_rate=0.001,
                           loss='mse',
                           metrics=[],
                           opt=None):

        # TODO: check input_shape form
        input_len = len(input_shape)
        assert input_len >= 2 and input_len <= 4, "CNN Input Shape must have dimensionality 2, 3, or 4"
    
        # TODO: check output_shape form

        activation = FullyConnectedNetwork.translate_activation_function(activation)
        activation_last = FullyConnectedNetwork.translate_activation_function(activation_last)

        # Convolution dimensionality
        conv_type = ['C1', 'C2', 'C3'][len(input_shape)-2]
        print("Convolution type:", conv_type)
    
        # Resolve regularizer
        regularizer = FullyConnectedNetwork.create_regularizer(lambda1, lambda2)

        # Input layer
        input_tensor = tensor = Input(shape=input_shape,
                                      name=name_base + 'input')
    
        # Dropout input features?
        #if dropout_input is not None:
        #tensor = Dropout(rate=dropout_input, name=name_base+"dropout_input")(tensor)
        tensor = ConvolutionalNeuralNetwork.create_cnn_stack(tensor,
                                                             conv_filters=conv_number_filters,
                                                             conv_kernel_size=conv_kernel_size,
                                                             conv_pool_average=conv_pool_average_size,
                                                             conv_pool=conv_pool_size,
                                                             conv_strides=conv_strides,
                                                             conv_type=conv_type,
                                                             name_base=name_base + '_C',
                                                             conv_activation=conv_activation,
                                                             regularizer=regularizer,
                                                             s_dropout=spatial_dropout,
                                                             padding=conv_padding,
                                                             batch_normalization=conv_batch_normalization,
                                                             global_max_pool=True)

        # Create the rest of the network
        tensor = FullyConnectedNetwork.create_dense_stack(tensor=tensor,
                                                          n_hidden=n_hidden,
                                                          activation=activation,
                                                          regularizer=regularizer,
                                                          dropout=dropout,
                                                          name=name_base+'_FC',
                                                          #name_last=name_last,
                                                          #activation_last=activation_last,
                                                          batch_normalization=batch_normalization)

        # Output layer
        if batch_normalization:
            tensor = BatchNormalization()(tensor)

        if len(output_shape) == 1:
            # Output is 1D
            tensor = Dense(output_shape[0],
                           use_bias=True,
                           bias_initializer='zeros',
                           name=name_base + 'output',
                           activation=activation_last,
                           kernel_regularizer=regularizer,
                           kernel_initializer='truncated_normal')(tensor)
        else:
            # Output is kD, where k > 1
            # Linear transformation from previous layer to one that is the product of
            #   all the output dimensions, then reshape, then non-linearity
            #
            #  This feature is useful for cases such as outputing K probability distributions for
            #   which we are using softmax as the non-linearity.
            
            tensor = Dense(reduce(operator.mul, output_shape, 1),
                           use_bias=True,
                           bias_initializer='zeros',
                           name=name_base + 'output_vector',
                           activation='linear',
                           kernel_regularizer=regularizer,
                           kernel_initializer='truncated_normal')(tensor)
            tensor = Reshape(output_shape)(tensor)
            tensor = Activation(activation_last)(tensor)


        if opt is None:
            opt = keras.optimizers.Adam(learning_rate=learning_rate,
                                        amsgrad=False)

        # Create the model
        model = Model(input_tensor, tensor)
    
        model.compile(loss=loss, optimizer=opt, metrics=metrics)

        return model
    
