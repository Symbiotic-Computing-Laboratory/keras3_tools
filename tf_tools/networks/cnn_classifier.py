'''
Andrew H. Fagg

CNN-based classifiers

'''
#print('CNN Classifier')

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import random

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Input, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, AveragePooling2D, Flatten, BatchNormalization, Dropout, Concatenate

#from tensorflow.keras.optimizers import RMSprop
#import random
import re

#from sklearn.p
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
                                  padding='valid'):
    model = Sequential()
    model.add(InputLayer(input_shape=(image_size[0], image_size[1], nchannels),
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
        
        if 'batch_normalization' in l and l['batch_normalization']:
            model.add(BatchNormalization())

        if l['pool_size'] is not None and l['pool_size'][0] > 1:
            model.add(MaxPooling2D(pool_size=l['pool_size'],
                               strides=l['strides'],
                               padding=padding))
          
        i=i+1
        
    # Flatten 
    model.add(Flatten())
    
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
                                  output_activation='softmax',
                                  loss='categorical_crossentropy',
                                  metrics=['categorical_accuracy'],
                                  padding='valid'):
    
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
                               padding=padding)
    # Output layer
    model.add(Dense(units=n_classes,
                    activation=output_activation,
                    use_bias=True,
                    bias_initializer='zeros',
                    name='%soutput'%(name_base)))
    
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=.9, beta_2=0.999,
                                  epsilon=None, decay=0.0, amsgrad=False)
    
    # Asssuming right now two outputs
    model.compile(loss=loss, optimizer=opt,
                 metrics=metrics) #, tf.keras.metrics.TruePositives(),
                         #tf.keras.metrics.FalsePositives(),
                         #tf.keras.metrics.TrueNegatives(),
                         #tf.keras.metrics.FalseNegatives()])
    return model

def training_set_generator_images(ins, outs, batch_size=10,
                          input_name='input', 
                        output_name='output'):
    '''
    Generator for producing random minibatches of image training samples.
    
    @param ins Full set of training set inputs (examples x row x col x chan)
    @param outs Corresponding set of sample (examples x nclasses)
    @param batch_size Number of samples for each minibatch
    @param input_name Name of the model layer that is used for the input of the model
    @param output_name Name of the model layer that is used for the output of the model
    '''
    
    while True:
        # Randomly select a set of example indices
        example_indices = random.choices(range(ins.shape[0]), k=batch_size)
        
        # The generator will produce a pair of return values: one for inputs and one for outputs
        if(len(outs.shape) == 1):
            yield({input_name: ins[example_indices,:,:,:]},
                  {output_name: outs[example_indices]})
        else:
            yield({input_name: ins[example_indices,:,:,:]},
                  {output_name: outs[example_indices,:]})

'''
TODO: finish
'''
def training_set_generator_k_images(ins, outs,
                                    n_images = 2,
                                    batch_size=10,
                                    input_name='input', 
                                    output_name='output'):
    '''
    Generator for producing random minibatches of image training samples.
    
    @param ins Full set of training set inputs (examples x row x col x chan)
    @param outs Corresponding set of sample (examples x nclasses)
    :param n_images: How many subsequent images in the input dictionary
    @param batch_size Number of samples for each minibatch
    @param input_name Name of the model layer that is used for the input of the model
    @param output_name Name of the model layer that is used for the output of the model
    '''
    
    while True:
        # Randomly select a set of example indices
        example_indices = random.choices(int(range(ins.shape[0]/n_images)), k=batch_size)

        # Assemble the output dictionary
        input_dict = {}
        for i in range(n_images):
            input_dict['input_name_%d'%i] = ins[n_images*example_indices+i,:,:,:]
        # The generator will produce a pair of return values: one for inputs and one for outputs
        yield(input_dict,
             {output_name: outs[n_images*example_indices,:]})


##########################################################################################
# Functions for building inception-style architectures

def create_path(name_base, tensor, path, nfilters, activation, regularizer):
    '''
    Create a single path in a parallel processing structure
    :param name_base: String name of this path
    :param tensor: Input tensor
    :param path: List of strings describing the steps in the path
    :param nfilters: Number of filters for the conv layers
    :param activation: Activation function for the conv layers
    :param regularizer: Regularizer
    :return: Output tensor
    '''
    for i, p in enumerate(path):
        if p == "c1":
            tensor = Convolution2D(filters=nfilters,
                                kernel_size=(1,1),
                                strides=(1,1),
                                padding='same',
                                use_bias=True,
                                kernel_initializer='truncated_normal',
                                bias_initializer='zeros',
                                name='%sC%d'%(name_base,i),
                                activation=activation,
                                kernel_regularizer=regularizer)(tensor)
        elif p == "c3":
            tensor = Convolution2D(filters=nfilters,
                                kernel_size=(3,3),
                                strides=(1,1),
                                padding='same',
                                use_bias=True,
                                kernel_initializer='truncated_normal',
                                bias_initializer='zeros',
                                name='%sC%d'%(name_base,i),
                                activation=activation,
                                kernel_regularizer=regularizer)(tensor)
        
        elif p == "c5":
            tensor = Convolution2D(filters=nfilters,
                                kernel_size=(5,5),
                                strides=(1,1),
                                padding='same',
                                use_bias=True,
                                kernel_initializer='truncated_normal',
                                bias_initializer='zeros',
                                name='%sC%d'%(name_base,i),
                                activation=activation,
                                kernel_regularizer=regularizer)(tensor)
        
        elif p == "mp2":
            tensor = MaxPooling2D(pool_size=(2,2),
                                  strides=(1,1),
                                  padding='same',
                                  name='%sMP%d'%(name_base,i))(tensor)
            
        elif p == "mp3":
            tensor = MaxPooling2D(pool_size=(3,3),
                                  strides=(1,1),
                                  padding='same',
                                  name='%sMP%d'%(name_base,i))(tensor)
        else:
            assert False, "Unknown path step: %s"%p

    return tensor
            
def create_module(name_base, tensor, paths, nfilters, activation, regularizer, dropout, compress_flag):
    '''
    Create a single inception-type module
    
    :param name_base: Base string name of the module
    :param tensor: input tensor
    :param paths: List of paths that make up this module
    :param nfilters: Scalar number of filters for the module
    :param activation: Activation function
    :param regularizer: Regularization function
    :param dropout: Dropout rate for spatial Dropout
    :param compress_flag: Indicate whether the module should compress rows and columns by a factor of 2
    :return: Output tensor
    
    '''
    
    outputs = []
    for i, p in enumerate(paths):
        outputs.append(create_path('%s_path_%d'%(name_base, i), tensor, p, nfilters, activation, regularizer))

    # Put all of the ouputs together
    if(len(outputs) > 1):
        tensor = Concatenate(axis=-1)(outputs)
    else:
        tensor = outputs[0]

    # Compress rows and columns by 2?
    if compress_flag:
        tensor = AveragePooling2D(pool_size=(2,2),
                                  strides=(2,2),
                                  padding='same',
                                  name='%sAP'%(name_base))(tensor)

    # Possible dropout
    if dropout is not None:
        tensor = SpatialDropout2D(dropout)(tensor)
        
    return tensor

def create_module_sequence(name_base, tensor, paths, nfilters, activation, regularizer, dropout, compress_flags):
    '''
    Create a sequence of inception-type modules
    
    :param name_base: Base string name of the module
    :param tensor: input tensor
    :param paths: List of paths that make up the modules
    :param nfilters: Array of filter numbers (one for each module)
    :param activation: Activation function
    :param regularizer: Regularization function
    :param compress_flags: Array of compress flags (one for each module)
    :return: Output tensor (flattened)
    
    '''
    
    for i, (n, c) in enumerate(zip(nfilters, compress_flags)):
        tensor = create_module('%s_module_%i'%(name_base,i), tensor, paths, n, activation, regularizer, dropout, c)

    tensor = Flatten()(tensor)
    
    return tensor

def create_parallel_module_sequences(name_base, tensor, paths, nfilters_list, activation, regularizer, dropout, compress_flags_list):
    '''
    Create a set of parallel module sequences and append them together
    '''
    output = []
    
    for i, (nfilters, compress_flags) in enumerate(zip(nfilters_list, compress_flags_list)):
        output.append(create_module_sequence('%s_ms_%d_'%(name_base,i), tensor, paths, nfilters, activation, regularizer,
                                             dropout, compress_flags))

    if(len(output) > 1):
        tensor = Concatenate(axis=-1)(output)
    else:
        tensor = output[0]
        
    return tensor

def create_inception_classifier_network(image_size, nchannels, 
                                        name_base='',
                                        preprocess_conv_layers=None,
                                        padding='same',
                                        paths=[],
                                        nfilters_list=[],
                                        compress_flags_list=[],
                                        conv_activation='elu',
                                        dense_layers=[],
                                        dense_activation='elu',
                                        n_classes=2, 
                                        lrate=.0001,
                                        regularizer=None,
                                        p_dropout=None,
                                        s_dropout=None,
                                        output_activation='softmax',
                                        loss='categorical_crossentropy',
                                        metrics=['categorical_accuracy']):

    #
    if isinstance(regularizer, (int, float)):
        regularizer=tf.keras.regularizers.l2(regularizer)
    

    # Create base model
    input_tensor = Input(shape=(image_size[0], image_size[1], nchannels),
                        name='%sinput'%(name_base))
    tensor = input_tensor

    if preprocess_conv_layers is not None:
        tensor = create_cnn_stack(tensor, preprocess_conv_layers,
                                  name_base='%s_conv_stack'%name_base,
                                  regularizer=regularizer,
                                  s_dropout=s_dropout,
                                  padding=padding)
        
    tensor = create_parallel_module_sequences('%s_inception'%name_base,
                                              tensor,
                                              paths,
                                              nfilters_list,
                                              conv_activation,
                                              regularizer,
                                              dropout=s_dropout,
                                              compress_flags_list=compress_flags_list)

    # Dense layers
    tensor = create_dense_stack(tensor=tensor,
                                n_hidden=dense_layers,
                                lambda_regularization=regularizer,
                                name=name_base,
                                activation=dense_activation,
                                dropout=p_dropout,
                                name_last='%s_D_out'%(name_base),
                                activation_last=dense_activation)

    
    # Output layer
    tensor = Dense(units=n_classes,
                    activation='softmax',
                    use_bias=True,
                    bias_initializer='zeros',
                    kernel_initializer='truncated_normal',
                    name='%s_output'%(name_base))(tensor)
    
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=.9, beta_2=0.999,
                                  epsilon=None, decay=0.0, amsgrad=False)

    # Create model
    model = Model(inputs=input_tensor, outputs=tensor)
    
    # Asssuming right now two outputs
    model.compile(loss=loss, optimizer=opt,
                 metrics=metrics)
    
    return model
