from tensorflow.keras import Input, Model, layers

def CNN_Model(input_shape, number_classes):
  # define Input layer
  input_tensor = Input(shape=input_shape) # Input: convert normal numpy to Tensor (float32)

  # define layer connection
  x = layers.Conv2D(filters = 32, kernel_size=(3, 3), activation="relu")(input_tensor)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu")(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu")(x)

  x = layers.Flatten()(x)
  #x = layers.Dropout(0.5)(x)
  x = layers.Dense(64, activation='relu')(x)
  outputs = layers.Dense(number_classes)(x) #, activation="softmax"

  # define model
  model = Model(inputs=input_tensor, outputs=outputs, name="TF_Tutorials_CNN_Model")
  return model