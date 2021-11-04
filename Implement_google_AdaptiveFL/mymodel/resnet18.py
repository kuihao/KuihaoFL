import tensorflow as tf
from tensorflow.keras.layers import (
  Conv2D,BatchNormalization,Activation,add,MaxPooling2D
)
from tensorflow.keras import Input, Model, layers, models, Sequential 

class myResNet():
  '''Method `ResNet18` is the keras function api
  implement the paper of arXiv:1512.03385
  returns: keras model'''
  def __init__(self):
    pass

  def __Conv2D_with_BN(self, input_tensor, FilterNum, FeatureMapSize, 
                    PaddingMode, StrideLength, name=None):
      '''#### The basic convolution layer:
      [arXiv:1512.03385, P4]
      We adopt batch normalization(BN)
      right after each convolution 
      '''
      x = Conv2D(filters=FilterNum,
                      kernel_size=FeatureMapSize,
                      padding=PaddingMode,
                      strides=StrideLength,
                      name=name)(input_tensor)
      x = BatchNormalization()(x)
      return x

  def __ResidualBlock(self, input_tensor, FilterNum, StrideLength, 
                    shortcut_shape_transform = False, name=None):
      '''Fully obedience to the paper for restoration'''
      x = self.__Conv2D_with_BN(input_tensor, FilterNum, FeatureMapSize=(3,3),
                        PaddingMode='same', StrideLength=StrideLength, name=name)
      x = Activation('relu')(x)
      x = self.__Conv2D_with_BN(x, FilterNum, FeatureMapSize=(3,3), 
                        PaddingMode='same', StrideLength=1)

      if shortcut_shape_transform:
          identity = self.__Conv2D_with_BN(input_tensor, FilterNum, FeatureMapSize=(1,1), 
                                    PaddingMode='valid', StrideLength=StrideLength)
          x = add([x,identity])
      else:
          x = add([x,input_tensor])
      
      x = Activation('relu')(x) # Why not `tf.nn.relu`? If you use it with Keras, you may face some problems while loading or saving the models or converting the model to TF Lite.
      return x

  def ResNet18(self, input_shape, ClassNum):
    '''Keras functional api version'''
    # Step 1. Create a input node
    inputs = Input(shape=input_shape, name="Input_layer")
    
    # Steop 2. Add more layers
    # Conv1
    x = self.__Conv2D_with_BN(inputs,64,(7,7),'same',2,name="Conv1")
    # Conv2
    x = MaxPooling2D(pool_size=(3,3), strides=1, padding='same',name="Conv2")(x)
    x = self.__ResidualBlock(x,64,1) #True?
    x = self.__ResidualBlock(x,64,1)
    # Conv3
    x = self.__ResidualBlock(x,128,2,True,name="Conv3")
    x = self.__ResidualBlock(x,128,1)
    # Conv4
    x = self.__ResidualBlock(x,256,2,True,name="Conv4")
    x = self.__ResidualBlock(x,256,1)
    # Conv5
    x = self.__ResidualBlock(x,512,2,True,name="Conv5")
    x = self.__ResidualBlock(x,512,1)
    # Average Pool 1x1
    x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(1000)(x) #,activation="relu" # toy
    #x = layers.Dropout(0.5)(x) # toy

    # Step 3. Create a output node
    outputs = layers.Dense(ClassNum, name="Output_layer")(x) #activation="softmax"

    # Step 4. Create a  Model to specifying its 
    # inputs and outputs in the graph of layers.
    return Model(inputs, outputs, name="resnet-18")

"""
# Keras Model Class
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

'''
# resnet18
# Auther: Dan
# Source-1: https://ithelp.ithome.com.tw/articles/10223034
# Source-2: https://colab.research.google.com/drive/1-drI94lTvh7-sn8nWMTptqhfXVNMzgSA 
'''
class ResBlock_Dan(layers.Layer):
  def __init__(self, filter_nums, strides=1, residual_path=False):
      super(ResBlock_Dan, self).__init__()

      self.conv_1 = layers.Conv2D(filter_nums,(3,3),strides=strides,padding='same')
      self.bn_1 = layers.BatchNormalization()
      self.act_relu = layers.Activation('relu')

      self.conv_2 = layers.Conv2D(filter_nums,(3,3),strides=1,padding='same')
      self.bn_2 = layers.BatchNormalization()
      
      if strides !=1:
        self.block = Sequential()
        self.block.add(layers.Conv2D(filter_nums,(1,1),strides=strides))
      else:
        self.block = lambda x:x


  def call(self, inputs, training=None):
      #call implement the model's forward pass
      x = self.conv_1(inputs)
      x = self.bn_1(x, training=training)
      x = self.act_relu(x)
      x = self.conv_2(x)
      x = self.bn_2(x,training=training)
      
      identity = self.block(inputs)
      outputs = layers.add([x,identity])
      outputs = tf.nn.relu(outputs)

      return outputs

class ResNet_Dan(Model):
  def __init__(self,layers_dims,nums_class):
    super(ResNet_Dan,self).__init__()

    self.model = Sequential([layers.Conv2D(64,(7,7),strides=(2,2), padding='same'),
                            layers.BatchNormalization(), #64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                            layers.Activation('relu'),
                            layers.MaxPooling2D(pool_size=(2,2),strides=(1,1),padding='same')]) 

    self.layer_1 = self.ResNet_build(64,layers_dims[0])
    self.layer_2 = self.ResNet_build(128,layers_dims[1],strides=2)   
    self.layer_3 = self.ResNet_build(256,layers_dims[2],strides=2) 
    self.layer_4 = self.ResNet_build(512,layers_dims[3],strides=2)   
    self.avg_pool = layers.GlobalAveragePooling2D()                 
    self.fc_model = layers.Dense(nums_class)    

  def call(self, inputs, training=None):
    x = self.model(inputs)
    x = self.layer_1(x)        
    x = self.layer_2(x) 
    x = self.layer_3(x)                               
    x = self.layer_4(x) 
    x = self.avg_pool(x) 
    x = self.fc_model(x)
    return x

  def ResNet_build(self,filter_nums,block_nums,strides=1):
    build_model = Sequential()
    build_model.add(ResBlock_Dan(filter_nums,strides))
    for _ in range(1,block_nums):
      build_model.add(ResBlock_Dan(filter_nums,strides=1))
    return build_model

def ResNet18_Dan(layers_dims=[2,2,2,2],nums_class=10):
  '''```python
    .build(input_shape=(None,?,?,?))
  ```'''
  return ResNet_Dan(layers_dims,nums_class)  

