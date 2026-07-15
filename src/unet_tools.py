'''
Luke Sewell, Andrew H. Fagg

U-Nets
'''

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import InputLayer, Input, BatchNormalization
from keras.layers import Dropout, Concatenate, Add
from keras.layers import Convolution1D, MaxPooling1D, SpatialDropout1D, UpSampling1D
from keras.layers import Convolution2D, MaxPooling2D, SpatialDropout2D, UpSampling2D
from keras.layers import Convolution3D, MaxPooling3D, SpatialDropout3D, UpSampling3D
from fully_connected_tools import *
from cnn_tools import *

class UNet:
    # Reference cnn_tools
    conv_operators = {'C1': (Convolution1D, MaxPooling1D, UpSampling1D, SpatialDropout1D),
                      'C2': (Convolution2D, MaxPooling2D, UpSampling2D, SpatialDropout2D),
                      'C3': (Convolution3D, MaxPooling3D, UpSampling3D, SpatialDropout3D),
                      }


    @staticmethod
    def create_unet_encoder_stack(tensor,
                                  unet_filters:[int],
                                  unet_kernel_size:[int],
                                  unet_pool:[int],
                                  conv_type:str='C2',
                                  name_base:str='',
                                  unet_activation:str='elu',
                                  regularizer=None,
                                  s_dropout=None,
                                  padding='same',
                                  batch_normalization=False):
        '''
        Create a linear convolutional stack
        
        '''
        # Access the layer constructors for the dimensionality of the input
        conv, mp, _, sd = UNet.conv_operators[conv_type]

        # Handle custom activation function strings
        unet_activation = FullyConnectedNetwork.translate_activation_function(unet_activation)


        # List for the skip connections
        skips = []

        
        # Loop over all convolutional layers
        for i, (f,k,p) in enumerate(zip(unet_filters, unet_kernel_size, unet_pool)):

            # Convolution
            tensor = conv(filters=f,
                          kernel_size=k,
                          strides=1,
                          padding=padding,
                          use_bias=True,
                          kernel_initializer='truncated_normal',
                          bias_initializer='zeros',
                          name='%s%d'%(name_base,i) + '_enc',
                          activation=unet_activation,
                          kernel_regularizer=regularizer)(tensor)
        
            # Possible dropout
            if s_dropout is not None and s_dropout > 0.0:
                tensor = sd(s_dropout)(tensor)

            # Batch Norm
            if batch_normalization:
                tensor = BatchNormalization()(tensor)

            # Downsample
            if p is not None and p > 1:
                # Save the feature map for the decoder, p > 1 prevents saving bottleneck as connection
                skips.append(tensor)
                tensor = mp(p,
                            padding=padding)(tensor)
            else: 
                skips.append(None)
            
        return tensor, skips

    @staticmethod
    def create_unet_decoder_stack(tensor,
                                  skips,
                                  unet_filters:[int],
                                  unet_kernel_size:[int],
                                  unet_pool:[int],
                                  conv_type:str='C2',
                                  skip_type='concat',
                                  name_base:str='',
                                  unet_activation:str='elu',
                                  regularizer=None,
                                  s_dropout=None,
                                  padding='same',
                                  batch_normalization=False
                                  ):

        # Access the layer constructors for the dimensionality of the input
        conv, mp, us, sd = UNet.conv_operators[conv_type]

        # Handle custom activation function strings
        unet_activation = FullyConnectedNetwork.translate_activation_function(unet_activation)
        
        # Skips are used in reverse order, last one sadded is first used
        # zip skips, filters, pool -> reverse -> enumerate
        for i, (skip, p, f, k) in enumerate(zip(reversed(skips), reversed(unet_pool), reversed(unet_filters), reversed(unet_kernel_size))):

            if skip is not None:
                # Upsample
                tensor = us(size=p, name='Up'+str(i))(tensor)
                
                # Add skip connection
                # Ensure tensor and skip are same shape for add
                if(skip_type == 'add'):
                    tensor = Add()([tensor, skip])
                elif(skip_type == 'concat'):
                    tensor = Concatenate()([tensor, skip])
                
            tensor = conv(filters=f,
                          strides=1,
                          kernel_size=k,
                          padding=padding, 
                          use_bias=True,
                          kernel_initializer='truncated_normal',
                          bias_initializer='zeros',
                          name=f"{name_base}{i}" + '_dec',
                          kernel_regularizer=regularizer,
                          activation=unet_activation)(tensor)
            
            if s_dropout is not None and s_dropout > 0:
                tensor = sd(s_dropout)(tensor)

            if batch_normalization:
                tensor = BatchNormalization()(tensor)     
                
        return tensor

    @staticmethod
    def create_unet(input_shape=None,
                    input_dtype=None,
                    unet_kernel_size=3,
                    unet_padding='same',
                    unet_number_filters=[8,16],
                    unet_pool_size=[None,2],
                    conv_kernel_size=3,
                    conv_padding='same',
                    conv_number_filters=[8,16],
                    conv_pool_size=[None,2],
                    spatial_dropout=None,
                    conv_batch_normalization=False,
                    skip_type='concat',
                    output_shape=[1],
                    name_base='',
                    unet_activation='elu',
                    conv_activation='elu',
                    lambda1:float=None,
                    lambda2:float=None,
                    name_last='output',
                    activation_last=None,
                    batch_normalization=False,
                    learning_rate=0.001,
                    loss='mse',
                    metrics:[str]=None,
                    metrics_weighted:[str]=None,
                    opt=None,
                    **kwargs):

        
        ##################
        # Less frequently used arguments are in kwargs
        
        # Infer input dtype
        if input_dtype is None:
            input_dtype = "float32"
        
        ##################

        input_len = len(input_shape)
        assert (input_len >= 2 and input_len <= 4), "U-Net Input Shape must have dimensionality 2, 3, or 4"

        # Check to make sure input is divisible by downsampling factor
        
        # Compute downsampling factor
        downsample = 1
        for p in unet_pool_size:
            if p is not None and p > 1:
                downsample *= p
        
        # Check every spatial dimension
        for dimension in input_shape[:-1]:
            if dimension is not None and dimension % downsample != 0:
                raise ValueError(
                    f"Input dimension {dimension} is not divisible by "
                    f"the downsampling factor ({downsample}).")


        # Convolution dimensionality
        conv_type = ['C1', 'C2', 'C3'][len(input_shape)-2]

        conv = UNet.conv_operators[conv_type][0]

            
        print("Dimension type:", conv_type)
    
        # Resolve regularizer
        regularizer = FullyConnectedNetwork.create_regularizer(lambda1, lambda2)

        # Input layer
        input_tensor = tensor = Input(shape=input_shape,
                                      name=name_base + '_input',
                                      dtype=input_dtype)

        
        # Create the downward part of the U
        tensor, skip_connections = UNet.create_unet_encoder_stack(tensor,
                                        unet_filters=unet_number_filters,
                                        unet_kernel_size=unet_kernel_size,
                                        unet_pool=unet_pool_size,
                                        conv_type=conv_type,
                                        name_base=name_base + '_C',
                                        unet_activation=unet_activation,
                                        regularizer=regularizer,
                                        s_dropout=spatial_dropout,
                                        padding=unet_padding,
                                        batch_normalization=conv_batch_normalization)

        # Upward part of the U
        tensor = UNet.create_unet_decoder_stack(tensor,
                                                skip_connections,
                                                skip_type=skip_type,
                                                unet_filters=unet_number_filters,
                                                unet_kernel_size=unet_kernel_size,
                                                unet_pool=unet_pool_size,
                                                conv_type=conv_type,
                                                name_base=name_base + '_C',
                                                unet_activation=unet_activation,
                                                regularizer=regularizer,
                                                s_dropout=spatial_dropout,
                                                padding=conv_padding,
                                                batch_normalization=conv_batch_normalization)

        # If user isn't specifying CNN arguments skip the cnn stack
        if conv_number_filters:
            tensor = ConvolutionalNeuralNetwork.create_cnn_stack(tensor,
                                                                 conv_filters=conv_number_filters,
                                                                 conv_kernel_size=conv_kernel_size,
                                                                 conv_pool_average=None,
                                                                 conv_pool=conv_pool_size,
                                                                 conv_strides=None,
                                                                 conv_type=conv_type,
                                                                 name_base=name_base,
                                                                 conv_activation=conv_activation,
                                                                 s_dropout=spatial_dropout,
                                                                 padding=conv_padding,
                                                                 batch_normalization=conv_batch_normalization)

        
        # Final output layer
        tensor = conv(filters=output_shape[-1],
                      kernel_size=1,
                      strides=1,
                      padding="same",
                      activation=activation_last,
                      name=name_last,
                     )(tensor)


        if opt is None:
            opt = keras.optimizers.Adam(learning_rate=learning_rate,
                                        amsgrad=False)

        # Create the model
        model = Model(input_tensor, tensor)
    
        model.compile(loss=loss, optimizer=opt, metrics=metrics, 
                      weighted_metrics=metrics_weighted)

        return model
