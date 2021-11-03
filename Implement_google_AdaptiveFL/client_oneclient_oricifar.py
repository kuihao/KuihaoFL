import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import flwr as fl
import tensorflow as tf
from tensorflow.keras import datasets, Input, Model, layers, models, Sequential
import numpy as np
import argparse
from mypkg.gpu.limitGPU import setGPU

# --------
# [limit the GPU usage]
# --------
setGPU(mode=1,gpus=tf.config.list_physical_devices('GPU'))

# --------
# [Hyperparemeter]
# --------
np.random.seed(2021)
tf.random.set_seed(2021)

# --------
# Step 1. Build Local Model (建立本地模型)
# --------
num_classes = 10
input_shape = (32, 32, 3)

class ResBlock(layers.Layer):
  def __init__(self, filter_nums, strides=1, residual_path=False):
      super(ResBlock, self).__init__()

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

      x = self.conv_1(inputs)
      x = self.bn_1(x, training=training)
      x = self.act_relu(x)
      x = self.conv_2(x)
      x = self.bn_2(x,training=training)
      
      identity = self.block(inputs)
      outputs = layers.add([x,identity])
      outputs = tf.nn.relu(outputs)

      return outputs

class ResNet(Model):
  def __init__(self,layers_dims,nums_class=10):
    super(ResNet,self).__init__()

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
    build_model.add(ResBlock(filter_nums,strides))
    for _ in range(1,block_nums):
      build_model.add(ResBlock(filter_nums,strides=1))
    return build_model

def ResNet18():
  return ResNet([2,2,2,2])

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

# --------
# Step 2. Load local dataset (引入本地端資料集)，對Dataset進行切割
# --------
def load_dataset(idx: int):
    x_train = np.load("EC10_IID/EC10_client_"+str(idx)+"_x.npy")
    y_train = np.load("EC10_IID/EC10_client_"+str(idx)+"_y.npy")
    x_train = x_train / 255.0
    y_train = tf.squeeze(y_train,axis=1)
    return (x_train,y_train)#, ()

# --------
# Step 3. Define Flower client 
# (定義client的相關設定: 接收 Server-side 的
#  global model weight、hyperparameters)
# --------
class MyClient(fl.client.NumPyClient):
    # Class初始化: local model、dataset
    def __init__(self, model, x_train, y_train, x_test, y_test, MachineID, SampleID):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.MachineID = MachineID
        self.SampleID = SampleID

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        #if self.MachineID > 5: # 無效，程式仍霸佔 GPU 記憶體，除非開第二張卡
        #  time.sleep(10)

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        rnd: int = config["rnd"]-1
        print("\n*** 本次為 round: ",config["rnd"]," ***\n") # 用 round 來對應抽樣列

        # 依據 rnd 對應 sample id 挑選對應 client 的 dataset
        #(self.x_train,self.y_train) = load_dataset(self.SampleID[rnd])

        # Train the model using hyperparameters from config
        # (依 Server-side 的 hyperparameters 進行訓練)
        history = self.model.fit(
            x = self.x_train,
            y = self.y_train,
            epochs = epochs,
            batch_size = batch_size,
        ) #validation_split=0.1, #batch_size, validation_data = (self.x_test, self.y_test)

        # Return updated model parameters and results
        # 將訓練後的權重、資料集筆數、正確率/loss值等，回傳至server-side
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            #"val_loss": history.history["val_loss"][0],
            #"val_accuracy": history.history["val_accuracy"][0]
            "top_k_categorical_accuracy": history.history["top_k_categorical_accuracy"][-1],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """K: 評估並非立即使用 local traing weight，而是使用新的 Global model weights"""
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(x = self.x_test, y = self.y_test, verbose = 2) # steps=steps
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}

'''
'''
# --------
# Step 4. Create an instance of our flower client and
# add one line to actually run this client. 
# (建立Client-to-Server的連線)
# --------
def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("-c", "--client", 
                        type=int, choices=range(0, 10), required=True,
                        help="from 0 to n")
    args = parser.parse_args()

    # Load and compile Keras model
    model = ResNet18()
    model.build(input_shape=(None,32,32,3))
    #model = CNN_Model(input_shape=(32, 32, 3), number_classes=10)
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    model.compile(optimizer, tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy", 'top_k_categorical_accuracy'])

    # Load a subset of CIFAR-10 to simulate the local data partition
    #SampleID = [[6, 8, 4, 3, 0, 7, 7, 9, 1, 5, 4, 4, 7, 1, 2, 2, 0, 1, 8, 4, 0, 8, 7, 1, 3, 9, 7, 4, 8, 5, 5, 6, 1, 3, 3, 7, 7, 7, 1, 8]]
    SampleID = [[51, 80, 69, 35, 31, 81, 4, 56, 60, 73, 8, 40, 34, 37, 60, 9, 20, 21, 6, 95, 13, 86, 91, 70, 37, 6, 68, 87, 59, 14, 25, 77, 81, 56, 
81, 35, 67, 43, 85, 47], [50, 88, 14, 28, 31, 59, 56, 60, 14, 82, 69, 28, 59, 60, 36, 17, 35, 0, 2, 68, 96, 29, 24, 96, 21, 31, 51, 54, 74, 83, 47, 88, 33, 6, 35, 9, 10, 22, 74, 57], [89, 28, 74, 17, 80, 54, 50, 99, 34, 55, 22, 80, 54, 17, 78, 88, 60, 36, 13, 44, 1, 57, 71, 37, 45, 80, 14, 9, 64, 
11, 55, 41, 74, 12, 92, 12, 20, 50, 20, 46], [26, 17, 10, 14, 3, 43, 94, 43, 69, 5, 71, 96, 35, 92, 3, 70, 74, 94, 87, 53, 22, 94, 14, 4, 0, 53, 43, 
4, 76, 51, 65, 61, 45, 69, 30, 32, 76, 65, 34, 42], [23, 55, 75, 44, 67, 48, 21, 68, 41, 8, 21, 29, 3, 77, 62, 64, 19, 74, 58, 84, 28, 43, 82, 70, 71, 20, 6, 43, 65, 23, 36, 23, 90, 45, 26, 88, 1, 47, 35, 7], [25, 52, 73, 10, 38, 21, 52, 65, 61, 7, 28, 99, 83, 40, 92, 95, 14, 58, 88, 19, 34, 42, 14, 52, 34, 94, 96, 28, 86, 75, 91, 82, 67, 5, 85, 38, 59, 99, 63, 36], [57, 51, 59, 37, 55, 90, 85, 67, 95, 68, 49, 1, 69, 17, 51, 68, 90, 32, 38, 46, 32, 34, 76, 29, 7, 34, 95, 97, 72, 32, 48, 13, 99, 12, 95, 2, 97, 80, 30, 51], [65, 35, 94, 82, 20, 57, 73, 74, 71, 29, 10, 81, 16, 35, 96, 72, 34, 60, 99, 86, 71, 35, 98, 52, 61, 99, 89, 75, 14, 64, 92, 97, 18, 17, 21, 42, 42, 22, 94, 27], [58, 44, 59, 94, 5, 16, 68, 19, 25, 46, 87, 46, 19, 92, 60, 28, 59, 16, 17, 2, 98, 58, 61, 22, 0, 0, 28, 56, 18, 64, 49, 50, 46, 81, 79, 83, 56, 6, 87, 10], [19, 24, 74, 29, 87, 2, 44, 7, 79, 26, 42, 17, 
38, 91, 93, 30, 73, 38, 61, 49, 26, 16, 55, 99, 80, 65, 0, 64, 77, 5, 57, 16, 79, 30, 63, 61, 65, 38, 17, 31]]
    
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    (x_train,y_train),_ = tf.keras.datasets.cifar10.load_data() 
    # Data preprocessing
    x_train = x_train / 255.0
    y_train = tf.squeeze(y_train,axis=1)
    #x_train = x_train / 255.0 # for cnn only

    # Start Flower client
    client = MyClient(model, x_train, y_train, '', '', args.client, '')
    #fl.client.start_numpy_client("localhost:8080", client=client) # windows
    fl.client.start_numpy_client("[::]:8080", client=client) # linux

# --------
# [Main]
# --------
if __name__ == "__main__":
    main()
