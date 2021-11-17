def myLoadDS(file_path='', ds_type='numpy'):
    repeat = True
    while(repeat):
        if ds_type =='numpy':
            repeat = False
            import numpy as np
            return np.load(file_path)
        elif ds_type =='tfds':
            repeat = False
            import tensorflow as tf
            return tf.data.experimental.load(file_path)
        else:
            print("Kuihso: Error!! Only accepy ds_type=numpy or tfds.")
