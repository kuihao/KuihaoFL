import tensorflow as tf
import numpy as np

#assert idx in range(10) # limit it can't more than 10...
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # Train 60000-5000, Test 10000
print(y_train[0])

# Data preprocessing
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1) # 這是讓輸入時圖片從 (28x28) 變成 28x28x1，
                                      # 即 -1 的維度從 28 array (一條像素) 
                                      # 變成 float32 (一個像素)，其實效果不大?
                                      # 主要是可對應彩色圖片是 3 通道，手寫辨識是單通道
x_test = np.expand_dims(x_test, -1) # 
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

def CNN_Model(input_shape, number_classes):
  # define tf.keras.Input layer
  input_tensor = tf.keras.Input(shape=input_shape) # tf.keras.Input: convert normal numpy to Tensor (float32)

  # define layer connection
  x = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3, 3), activation="relu")(input_tensor)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu")(x)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  outputs = tf.keras.layers.Dense(number_classes)(x) #activation="softmax"

  # define model
  model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="mnist_model")
  return model

'''
model = CNN_Model(input_shape=(28, 28, 1), number_classes=10)
model.summary()
model.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

history = model.fit(
            x_train,
            y_train,
            128,
            5,
            validation_data=(x_test,y_test),
        )
'''
