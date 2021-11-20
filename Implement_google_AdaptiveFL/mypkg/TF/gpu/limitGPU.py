#import os 
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

def setGPU(mode = 1, gpus = tf.config.list_physical_devices('GPU'), device_num=0):
    '''
    mode 1: which attempts to allocate only as much GPU memory as needed for the runtime allocations.
    mode 2: set a hard limit (1 GB) on the total memory to allocate on the GPU.
    mode 3: (Abandoned, not work) choose GPU device to run
    To see more info.: https://www.tensorflow.org/guide/gpu
    '''
    if mode == 1 and gpus:
        try:
            number_gpu = len(gpus)
            if number_gpu > 1:
                # 當有多張顯卡，但不設定可視範圍，則無法直接設定 memory limit
                # 會報 Memory growth cannot differ between GPU devices
                # 此處直接可見所有卡，但預設似乎都先跑到第0張
                tf.config.set_visible_devices(gpus[0:number_gpu-1], 'GPU')
            
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print("Kuihao:", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print("Kuihao:", e)
    elif mode == 2 and gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print("Kuihao:", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print("Kuihao:", e)
    elif mode == 3 and gpus: # 同時使用雙GPU的設定
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print("Kuihao:", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            '''
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print("Kuihao:", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            '''
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print("Kuihao:", e)
        
    elif not(gpus):
        raise Exception("Kuihao: There is no GPU allowed,"\
            " check if the program turned off the GPU access.")
    else:
        raise ValueError('Kuihao: Only accept 1 or 2. or sth else error!')
    
    print("Kuihao: GPUs are ready! :D\n")

#setGPU()





