from tensorflow.keras import Input, Model, layers

def CNN_Model(input_shape, ClassNum):
  # Step 1. Create a input node
  input_tensor = Input(shape=input_shape)
  '''Input: convert normal numpy to Tensor (float32)'''

  # Steop 2. Add more layers
  x = layers.Conv2D(filters = 32, kernel_size=(3, 3), activation="relu")(input_tensor)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu")(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu")(x)

  x = layers.Flatten()(x)
  #x = layers.Dropout(0.5)(x)
  x = layers.Dense(64, activation='relu')(x)
  
  # Step 3. Create a output node
  outputs = layers.Dense(ClassNum)(x) #, activation="softmax"

  # Step 4. Create a  Model to specifying its 
  # inputs and outputs in the graph of layers.
  model = Model(inputs=input_tensor, outputs=outputs, name="TF_Tutorials_CNN_Model")
  return model