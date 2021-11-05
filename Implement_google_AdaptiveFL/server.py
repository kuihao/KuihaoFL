import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np 
import flwr as fl
from flwr.server.strategy import FedAvg, FedYogi, FedAdam, FedAdagrad
from mypkg import (
    ServerArg, 
    ModelNameGenerator,
    setGPU,
    secure_mkdir,
    MyFedAdagrad,
    MyFedYogi,
    MyFedAdam
)
from mymodel import CNN_Model, myResNet

# --------
# [Hyperparemeter]
# --------
SEED = 2021
'''fix random seed'''
SAVE = False
'''(bool) save log or not'''
model_input_shape = (32,32,3)
model_class_number = 10
HyperSet_Model = CNN_Model(model_input_shape,model_class_number)
#CNN_Model(model_input_shape,model_class_number)
#myResNet().ResNet18(model_input_shape,model_class_number)
HyperSet_Aggregation = FedAvg

# --------
# [Global varables]
# --------
Training_result_distributed = {'loss':[],'accuracy':[],'top_k_categorical_accuracy':[]}
'''Clients Training 的聚合結果'''
Testing_result_distributed = {'loss':[],'accuracy':[],'top_k_categorical_accuracy':[]}
'''[未使用] Clients Testing 的聚合結果'''
Training_result_centralized = {'loss':[],'accuracy':[],'top_k_categorical_accuracy':[]}
'''[未使用] Server Training 的結果'''
Testing_result_centralized = {'loss':[],'accuracy':[],'top_k_categorical_accuracy':[]}
'''Server Testing 的結果'''

# --------
# [Welcome prompt] Make model name
# --------
args = ServerArg()
model_name = ModelNameGenerator(args.name)
print("This model name:", model_name)

# --------
# [Hardware setting] CPU only or limit the GPU usage
# --------
if args.cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    setGPU(mode=2) # gpus=tf.config.list_physical_devices('GPU')

import tensorflow as tf
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------
# [Main]
# --------
def main() -> None:
    # --------
    # [待修改] 建立 Global model、設定 FL 策略、啟動 Server、儲存 FL 結果
    # --------
    # Step 1. Build Global Model (建立全域模型)
    model = HyperSet_Model
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    model.compile(optimizer, 
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=["accuracy", 'top_k_categorical_accuracy'])

    # Step 2. Make the strategy (制定聯合學習策略)
    strategy = MyAggregation(
        fraction_fit=1.0, # 每一輪參與Training的Client比例
        #fraction_eval=1.0, # 每一輪參與Evaluating的Client比例
        min_fit_clients=1, # 每一輪參與Training的最少Client連線數量 (與比例衝突時,以此為準)
        #min_eval_clients=3, # 每一輪參與Evaluating的最少Client連線數量 (與比例衝突時,以此為準)
        min_available_clients=1, # 啟動聯合學習之前，Client連線的最小數量
        
        on_fit_config_fn=fit_config, # 設定 Client-side Training Hyperparameter  
        on_evaluate_config_fn=None, #evaluate_config, # 設定 Client-side Evaluating Hyperparameter
        eval_fn=get_eval_fn(model), # 設定 Server-side Evaluating Hyperparameter (用Global Dataset進行評估)
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()), # Global Model 初始參數設定
    )

    # Step 3. Run the server with the strategy
    #fl.server.start_server("localhost:8080", config={"num_rounds": 3}, strategy=strategy) #windows
    fl.server.start_server("[::]:8080", config={"num_rounds": 40}, strategy=strategy) #linux

    # [Kuihao addition] Save FL results to numpy-zip
    global Training_result_distributed, Testing_result_centralized
    if SAVE:
        FL_Results_folder = secure_mkdir("FL_Results"+"/"+model_name)
        np.savez(f"{FL_Results_folder}/Training_result_distributed.npz", Training_result_distributed)
        np.savez(f"{FL_Results_folder}/Testing_result_centralized.npz", Testing_result_centralized)

# --------
# [Customized Aggregation Strategy]
# --------
class MyAggregation(HyperSet_Aggregation):
    '''
    (1) 制定聚合演算法 (FedAvg MyFedAdagrad MyFedAdam MyFedYogi)\n
    (2) 保存Clients的Training結果\n
    (3) 儲存Global model weights
    '''
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        '''Override'''
        # Call Parent-class's aggregate_fit()
        aggregated_weights = super().aggregate_fit(rnd, results, failures) 

        # Aggregate clients' training results (loss, acc., top-k-acc.)
        examples = [r.num_examples for _, r in results]
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        accuracy_aggregated = sum(accuracies) / sum(examples)
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        loss_aggregated = sum(losses) / sum(examples)
        topK_accuracies = [r.metrics["top_k_categorical_accuracy"] * r.num_examples for _, r in results]
        topK_accuracies_aggregated = sum(topK_accuracies) / sum(examples)
        
        # [Kuihao addition] 暫存Client-side訓練結果
        global Training_result_distributed
        Training_result_distributed["loss"].append(loss_aggregated)
        Training_result_distributed["accuracy"].append(accuracy_aggregated)
        Training_result_distributed["top_k_categorical_accuracy"].append(topK_accuracies_aggregated)
        print(f"\n****\nRound {rnd}, train results from clients after aggregation:\n"\
        f"Loss:{loss_aggregated} Acc.:{accuracy_aggregated} TopK Acc.:{topK_accuracies_aggregated}"\
        f"\n****")

        # [Kuihao addition] 保存每一輪 Global Model Weights
        if SAVE:
            checkpoint_folder = secure_mkdir("ckpoint"+"/"+model_name)
            if aggregated_weights is not None:
                # Save aggregated_weights
                print(f"****Saving round {rnd} aggregated_weights...****")
                np.savez(f"{checkpoint_folder}/round-{rnd}-weights.npz", *aggregated_weights)

        return aggregated_weights

# --------
# [Server-side function] config and evaluate
# --------
def fit_config(rnd: int):
    """
    [Client-side, training hyperparameter]
    * 設定Client Training 的 Hyperparameter: 包含batch_size、epochs、
      learning-rate...皆可設定。
    * 甚至可以設定不同 FL round 給予 client 不同的 Hyperparameter
    * Return training configuration dict for each round.
      Keep batch size fixed at 128, perform two rounds of training with one
      local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1,
        "rnd":rnd,
    }
    return config

def evaluate_config(rnd: int):
    """
    [Client-side, evaluating hyperparameter]
    * 設定Client Testing 的 Hyperparameter: 包含epochs、steps(Total number of steps, 
      也就是 batche個數 (batches of samples))。
    * 可以設定不同 FL round 給予 client 不同的 Hyperparameter
    * Return evaluation configuration dict for each round.
      Perform five local evaluation steps on each client (i.e., use five
      batches) during rounds one to three, then increase to ten local
      evaluation steps.
    """
    val_steps = 1 # not use
    return {"val_steps": val_steps}

def get_eval_fn(model):
    '''
    [Server-side, evaluating hyperparameter]
    * Return an evaluation function for server-side evaluation.
    * 用 Global Dataset 評估 Global model (不含訓練)
    '''
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    _ ,(x_test,y_test) = tf.keras.datasets.cifar10.load_data()

    # Data preprocessing
    x_test = x_test / 255.0
    y_test = tf.squeeze(y_test,axis=1) # cifar10 version

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy,top_k_categorical_accuracy = model.evaluate(x_test, y_test)
        
        # [Kuihao addition] 暫存Server-side評估結果
        global Testing_result_centralized
        Testing_result_centralized["loss"].append(loss)
        Testing_result_centralized["accuracy"].append(accuracy)
        Testing_result_centralized["top_k_categorical_accuracy"].append(top_k_categorical_accuracy)

        return loss, {"accuracy": accuracy, 'top_k_categorical_accuracy':top_k_categorical_accuracy}

    return evaluate

# --------
# [Main Exe.]
# --------
if __name__ == "__main__":
    main()