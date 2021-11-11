import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import flwr as fl
from mypkg import ClientArg, FixClientSample, DynamicClientSample

# --------
# [Commandline Args]
# --------
args = ClientArg()

# --------
# [Hardware setting] CPU only or limit the GPU usage
# --------
if args.cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    #from mypkg.TF import setGPU
    #setGPU(mode=1)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
        from mypkg.TF import setGPU
        setGPU(mode=3, device_num=args.gpu)
    else:
        from mypkg.TF import setGPU
        setGPU(mode=1) # Dataset size 會影響 GPU memory 需求
import tensorflow as tf
from mypkg.TF import CNN_Model, myResNet

# --------
# [Hyperparemeter]
# --------
SEED = 2021
np.random.seed(SEED)
tf.random.set_seed(SEED)
'''fix random seed'''
model_input_shape = (32,32,3)
model_class_number = 100 # This is LABEL 
HyperSet_Model = myResNet().ResNet18(model_input_shape,model_class_number)
#CNN_Model(model_input_shape,model_class_number)
#myResNet().ResNet18(model_input_shape,model_class_number)
HyperSet_SampleRound = 10
HyperSet_SampleRange = 500

# --------
# [Main]
# --------
def main() -> None:
    # Build Local Model (建立本地模型)
    model = HyperSet_Model
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    model.compile(optimizer, 
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=["accuracy", 'top_k_categorical_accuracy'])

    # Start Flower client

    ## single client wuth original dataset test
    #(x_train,y_train) = load_cifar10()
    #(x_train,y_train) = load_EC10_client10
    #client = MyClient(model, x_train, y_train, '', '', args.client, SampleIDs[args.client])
    
    ## old version
    #SampleIDs = FixClientSample(100)
    #client = MyClient(model, '', '', '', '', args.client, SampleIDs[args.client])
    
    ## new auto sampling version
    SampleIDs = DynamicClientSample(rounds=HyperSet_SampleRound,
                                    client_range=HyperSet_SampleRange,
                                    client_id=args.client,
                                    )
    print(SampleIDs)
    client = MyClient(model, '', '', '', '', args.client, SampleIDs)
    #fl.client.start_numpy_client("localhost:8080", client=client) # windows
    fl.client.start_numpy_client("[::]:8080", client=client) # linux

# --------
# [Load local dataset]
# --------
def load_EC10(idx: int):
    x_train = np.load("EC10_IID/EC10_client_"+str(idx)+"_x.npy")
    y_train = np.load("EC10_IID/EC10_client_"+str(idx)+"_y.npy")
    x_train = x_train / 255.0
    if len(y_train.shape)>1:
        y_train = tf.squeeze(y_train,axis=1)
    return (x_train,y_train)
"""
def load_EC10_client10(idx: int):
    '''dataset size = 50k/client'''
    x_train = np.load("dataset/ec_x_"+str(idx)+".npy")
    y_train = np.load("dataset/ec_y_"+str(idx)+".npy")
    x_train = x_train / 255.0
    if len(y_train.shape)>1:
        y_train = tf.squeeze(y_train,axis=1)
    return (x_train,y_train)
"""
def load_cifar10():
    (x_train,y_train), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    if len(y_train.shape)>1:
        y_train = tf.squeeze(y_train,axis=1)
    return (x_train,y_train)

# --------
# [Define Flower Client]
# --------
class MyClient(fl.client.NumPyClient):
    # Class初始化: local model、dataset
    def __init__(self, model, x_train, y_train, x_test, y_test, MachineID, SampleID):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.MachineID = MachineID
        self.SampleID = SampleID
        self.num_examples_train = 0

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # Update local model parameters
        
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        rnd: int = config["rnd"]-1
        print("\n*** 本次為 round: ",config["rnd"]," ***\n") # Use round to pair sampling IDs

        # 依據 rnd 對應 sample id 挑選對應 client 的 dataset
        #(self.x_train,self.y_train) = load_EC10(self.SampleID[rnd])
        tfds_train = tf.data.experimental.load(f'dataset/cifar100_client_train/content/zip/cifar100_client/train/client_{self.SampleID[rnd]}_train')
        # size of dataset
        self.num_examples_train = sum(1 for _ in tfds_train)

        # Train the model using hyperparameters from config
        # (依 Server-side 的 hyperparameters 進行訓練)
        history = self.model.fit(
            #x = self.x_train,
            #y = self.y_train,
            tfds_train,
            epochs = epochs,
            batch_size = batch_size,
        ) #validation_split=0.1, #batch_size, 
        #validation_data = (self.x_test, self.y_test)

        # Return updated model parameters and results
        # 將訓練後的權重、資料集筆數、正確率/loss值等，回傳至server-side
        parameters_prime = self.model.get_weights()
        results = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            #"val_loss": history.history["val_loss"][0],
            #"val_accuracy": history.history["val_accuracy"][0]
            "top_k_categorical_accuracy": history.history["top_k_categorical_accuracy"][-1],
        }
        #tf.keras.backend.clear_session()
        return parameters_prime, self.num_examples_train, results

    def evaluate(self, parameters, config):
        """Kuihao: 評估並非立即使用 local traing weight，而是使用新的 Global model weights"""
        """Evaluate parameters on the locally held test set."""
        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(x = self.x_test, y = self.y_test, verbose = 2) # steps=steps
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}

# --------
# [Main Exe.]
# --------
if __name__ == "__main__":
    main()
