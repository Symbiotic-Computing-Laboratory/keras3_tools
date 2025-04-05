def str_to_bool(string):
    ''' 
    Converts a string to a boolean value.
    
    If the string contains the word "True" then the function returns
    a boolean value of True. Anything else will return a value of False.
    This is case insensitive.
    
    Args: string (str) 
    Returns:
      A boolean value of True or False
    '''
    

    if string.lower() == 'true':
        return True 

    return False


def generate_conv_layers(size, nfilters, pool, batch_norm=None):

    # This if statement was added for backwards compatibility. 
    if batch_norm is None: 
        return [{'filters': f, 'kernel_size': (s,s),
                    'pool_size': (p,p), 'strides': (p,p), 'batch_normalization': False} if p > 1
                else {'filters': f, 'kernel_size': (s,s),
                        'pool_size': None, 'strides': (1, 1), 'batch_normalization': False}
                for s, f, p in zip(size, nfilters, pool)]

    return [{'filters': f, 'kernel_size': (s,s),
                    'pool_size': (p,p), 'strides': (p,p), 'batch_normalization': str_to_bool(bn)} if p > 1
                else {'filters': f, 'kernel_size': (s,s),
                        'pool_size': None, 'strides': (1, 1), 'batch_normalization': str_to_bool(bn)}
                for s, f, p, bn in zip(size, nfilters, pool, batch_norm)]
