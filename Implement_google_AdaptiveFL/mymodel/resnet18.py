'''
* resnet18 (keras functional api version)
* Code auther: Kuihao Date: 2021/11/04
* Implement the paper of arXiv:1512.03385
'''
from tensorflow.keras.layers import (
  Conv2D,BatchNormalization,Activation,add,MaxPooling2D
)
from tensorflow.keras import Input, Model, layers

class myResNet():
  '''
  Method: 
  * `ResNet18(self, input_shape, ClassNum)` to generate a resnet18 keras model
  * keras function api version
  * implement the paper of arXiv:1512.03385
  '''
  def __init__(self):
    pass

  def __Conv2D_with_BN(self, input_tensor, FilterNum, FeatureMapSize, 
                    PaddingMode, StrideLength, name=None):
      '''The basic convolution layer:\n 
      We adopt batch normalization(BN)
      right after each convolution (arXiv:1512.03385, P4) 
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
    '''
    Args:
    * `input_shape`: tuple, without None (batch size)
    * `ClassNum`: int, number of label classes (output shape)
    
    Returns: 
    * keras model
    '''
    # Step 1. Create a input node
    inputs = Input(shape=input_shape, name="Input_layer")
    
    # Steop 2. Add more layers
    # Conv1
    x = self.__Conv2D_with_BN(inputs,64,(7,7),'same',2,name="Conv1")
    # Conv2
    x = MaxPooling2D(pool_size=(3,3), strides=1, padding='same',name="Conv2")(x)
    x = self.__ResidualBlock(x,64,1) #True? no, shape not change
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
    #x = layers.Dense(1000)(x) #,activation="relu" # tf-toy resnet
    #x = layers.Dropout(0.5)(x) # tf-toy resnet
    #x = layers.Flatten()(x) # paper
	  #x = layers.Dense(ClassNum,activation='softmax')(x) # paper

    # Step 3. Create a output node
    outputs = layers.Dense(ClassNum, name="Output_layer")(x) #activation="softmax"

    # Step 4. Create a  Model to specifying its 
    # inputs and outputs in the graph of layers.
    return Model(inputs, outputs, name="resnet-18")