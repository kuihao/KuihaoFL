'''
# resnet18
# Auther: Dan
# Source-1: https://ithelp.ithome.com.tw/articles/10223034
# Source-2: https://colab.research.google.com/drive/1-drI94lTvh7-sn8nWMTptqhfXVNMzgSA 
'''
import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential 
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