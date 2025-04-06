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
    
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    #beta_1=.9, beta_2=0.999,
    #epsilon=None, decay=0.0, amsgrad=False)

    # Create model
    model = Model(inputs=input_tensor, outputs=tensor)
    
    # Asssuming right now two outputs
    model.compile(loss=loss, optimizer=opt,
                 metrics=metrics)
    
    return model
