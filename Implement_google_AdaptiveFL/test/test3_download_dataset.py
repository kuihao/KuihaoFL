import tensorflow as tf

(x_train,y_train) ,(x_test,y_test) = tf.keras.datasets.cifar10.load_data() 

print(x_train.shape)
print(y_train.shape)

(x_train,y_train) ,(x_test,y_test) = tf.keras.datasets.mnist.load_data() 

print(x_train.shape)
print(y_train.shape)