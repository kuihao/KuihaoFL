from tensorflow.keras import Input, Model, layers
"""
# Keras Model Class style
  class ResNet18(tf.keras.Model):
    def __init__(self):
      super(ResNet18, self).__init__()
      self.conv1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs):
      x = self.dense1(inputs)
      return self.dense2(x)
"""
def toy_resnet():
    ''' 
    Keras functional api
        toy_resnet (keras functoinal api)
        https://www.tensorflow.org/guide/keras/functional#a_toy_resnet_model
    '''
    # Step 1. Create a input node
    inputs = Input(shape=(32, 32, 3), name="img")
    ''' Use `inputs.shape` to get the TensorShape.
        ```python
        inputs.shape
        ```
    '''
    # Steop 2. Add more layers
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # Step 3. Create a output node
    outputs = layers.Dense(10)(x)

    # Step 4. Create a  Model to specifying its 
    # inputs and outputs in the graph of layers.
    model = Model(inputs, outputs, name="toy_resnet")

    # Check out what the model summary looks lik
    #model.summary()
    # You can also plot the model as a graph
    #keras.utils.plot_model(model, "my_first_model.png") 
    return model