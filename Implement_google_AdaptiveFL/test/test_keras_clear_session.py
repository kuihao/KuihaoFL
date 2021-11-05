import time
import tensorflow as tf
from mypkg.TF import CNN_Model, myResNet, setGPU

setGPU()

model_input_shape = (32,32,3)
model_class_number = 10
for i in range(200):
    #tf.keras.backend.clear_session()
    #(x_train,y_train), _ = tf.keras.datasets.cifar10.load_data()
    model = CNN_Model(model_input_shape,model_class_number)
    print(i,model.name)
    time.sleep(0.1)
