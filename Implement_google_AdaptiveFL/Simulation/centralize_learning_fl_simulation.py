# -*- coding: utf-8 -*-
"""Kuihao_FL_Sequential_Multi-Clients_Simulation

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/kuihao/2b8b376fd307d83661fcd65679cc99ec/kuihao_fl_sequential_multi-clients_simulation.ipynb

#Prepare Dataset
"""

dataset_path = 'dataset/cifar100_noniid/content/zip/cifar100_noniid'
#dataset_path = r'C:\Users\kuiha\OneDrive - 國立成功大學 National Cheng Kung University\NCKU研究所\FL論文andCode\FlowerFL_code\實驗資料集\content\zip\cifar100_noniid'

"""#IMPORT PKG"""

import os

from tensorflow.python.saved_model.loader_impl import parse_saved_model
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
# add path to use my package
#sys.path.append('/Implement_FedAdativate')#/home/sheng/document/Kuihao
from datetime import datetime
import numpy as np

from mypkg import (
    ServerArg, 
    ModelNameGenerator,
    secure_mkdir,
    mylog,
    Result_Avg_Last_N_round,
    Simulation_DynamicClientSample,
    Weighted_Aggregate,
    FedAdagrad_Aggregate,
    FedAdam_Aggregate,
    FedYogi_Aggregate,
)

#import tensorflow as tf
'''
from mypkg.TF import (
    CNN_Model, 
    myResNet, 
    GoogleAdaptive_tfds_preprocessor, 
    simple_cifar100_preprocessor,
    myLoadDS
    )
'''

"""# Desktop Setting"""

# --------
# [Welcome prompt] Make model name
# --------
args = ServerArg()
model_name = ModelNameGenerator(args.name)
print(f"*** This model name: {model_name} ***\n")

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

# --------
# [Model Recovery Setting]
# --------
checkpoint_load = None
prior_model = None
if args.mode == 1:
    '''The prior temporary checkpoint was interrupt accidentally.'''
    prior_model = args.prior_model_name  
elif args.mode == 2:
    '''Load the other model train finished.'''
    checkpoint_load = args.checkpoint_path

import tensorflow as tf
import tensorflow_addons as tfa
from mypkg.TF import (
    CNN_Model, 
    myResNet, 
    GoogleAdaptive_tfds_preprocessor, 
    simple_cifar100_preprocessor,
    myLoadDS
)

"""#[Hyperparemeter]"""

#model_name = 'FL_Simulattion'
SEED = 2021
'''fix random seed'''
np.random.seed(SEED)
tf.random.set_seed(SEED)
model_input_shape = (24,24,3)
model_class_number = 100 # This is LABEL 

SAVE = True
'''(bool) save log or not'''
HyperSet_Aggregation, Aggregation_name = '', '' #Weighted_Aggregate
HyperSet_round = 600 # 4000*10 / 500 = 80
HyperSet_Train_all_connect_client_number = 500
HypHyperSet_Train_EveryRound_client_number = 500
HyperSet_Test_all_connect_client_number = 100
HypHyperSet_Test_EveryRound_client_number = 100

HyperSet_Server_eta = pow(10,(0)) #1e-3
HyperSet_Server_tau = None #pow(10,(-1)) #1e-2
HyperSet_Server_beta1 = None #0.9 
HyperSet_Server_beta2 = None #0.99

HyperSet_Local_eta = None #pow(10,(-1/2)) #1e-1
'''Don't use this'''
HyperSet_Local_momentum = 0. #0.9
HyperSet_Local_batch_size = 20
HyperSet_Local_epoch = None

HyperSet_optimizer = tf.keras.optimizers.SGD(learning_rate=HyperSet_Server_eta, momentum=HyperSet_Local_momentum)
'''
optimizer = tfa.optimizers.Yogi(learning_rate=HyperSet_Server_eta, 
                                epsilon=HyperSet_Server_tau,
                                beta1=HyperSet_Server_beta1,
                                beta2=HyperSet_Server_beta2,
                                )'''
#optimizer = tf.keras.optimizers.SGD(learning_rate=HyperSet_Local_eta, momentum=HyperSet_Local_momentum)
#optimizer = tf.keras.optimizers.Adam() #learning_rate=1e-5

# ---
# [Build-Model]
# ---
tf.keras.backend.clear_session()
model = myResNet().ResNet18(model_input_shape,model_class_number)
optimizer = HyperSet_optimizer
model.compile( optimizer, 
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=["accuracy", 'sparse_top_k_categorical_accuracy']) # sparse_top_k_categorical_accuracy, top_k_categorical_accuracy

# ---
# [Preprocessing Setting]
# ---
# Random number generator
rng = tf.random.Generator.from_seed(110, alg='philox')
preprocessor = GoogleAdaptive_tfds_preprocessor(
                          global_seed=SEED, 
                          crop_size=24, 
                          batch_zize=HyperSet_Local_batch_size, 
                          shuffle_buffer=100, 
                          prefetch_buffer=20,              
                        )

# --------
# [Saving Setting]
# --------
Training_result_distributed = {'loss':[],'accuracy':[],'sparse_top_k_categorical_accuracy':[]}
'''Clients Training 的聚合結果'''
Testing_result_centralized = {'loss':[],'accuracy':[],'sparse_top_k_categorical_accuracy':[]}
'''Server Testing 的結果'''
checkpoint_folder = secure_mkdir("ckpoint"+"/"+model_name)
'''保存weight的資料夾'''
checkpoint_path = checkpoint_folder+"/cp-{epoch:04d}.ckpt"
'''保存weight的儲存路徑'''
dataset_size = 50000
batch_counts_per_epoch = int(dataset_size/HyperSet_Local_batch_size)
'''steps_per_execution'''
cp_saver = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=2,
                                                 save_freq=100*batch_counts_per_epoch)
'''儲存機制: 開啟新的Epoch時(訓練前)先儲存，因此cp-0001其實是 epoch == 0的weights
save_freq: (int) the callback saves the model at end of this many batches.'''

#if prior_model is not None:
#    tmpbackup_folder = secure_mkdir("/tmp/backup"+"/"+prior_model)
#    '''暫存cp的資料夾'''
#    cp_recovery = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=tmpbackup_folder)
#else:
#    tmpbackup_folder = secure_mkdir("/tmp/backup"+"/"+model_name)
#    '''暫存cp的資料夾'''
#    cp_recovery = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=tmpbackup_folder)
#'''Store last cp in the tmp forder'''

if prior_model is not None:
    '''cpbackup暫時有bug，因此改用npz暫存檔代替其功能'''
    weight_npzfile = np.load(f'ckpoint/{prior_model}/interrupt-round-weights.npz')
    weight_np_unzip = [weight_npzfile[ArrayName] for ArrayName in weight_npzfile.files]
    '''model weights format is *List of NumPy arrays* '''
    model.set_weights(weight_np_unzip) # mot use model.load_weights() that's load from file.h5 

TraceCurrentEpoch = 0
class CustomCallback(tf.keras.callbacks.Callback):
    '''自訂Callback事件: 此用來追蹤當前執行Epoch
    Ref.:https://ithelp.ithome.com.tw/articles/10235293'''
    def __init__(self):
        self.task_type=''
        self.epoch=0
        self.batch=0
    def on_epoch_begin(self, epoch, logs=None):
        #print(f"{self.task_type}第 {epoch} 執行週期結束.")
        # Get epoch  now
        global TraceCurrentEpoch
        TraceCurrentEpoch = epoch
        #print(TraceCurrentEpoch)
    
# ---
# [Load Data]
# ---
# Load Server-side test daraset
server_train_data = myLoadDS(dataset_path+'/server/train/global_train_all', 'tfds')
#server_train_data = myLoadDS(dataset_path+f'/client/train/client_{1}_train', 'tfds')
server_test_data = myLoadDS(dataset_path+'/server/test/global_test_all', 'tfds')

"""
#Centralized
"""
try:
  tf.keras.backend.clear_session() #clear keras tmp data
  if checkpoint_load is not None:
    '''load model weights'''
    latest = tf.train.latest_checkpoint(checkpoint_load)
    model.load_weights(latest)
    # If load weight.npz
    #weight_npzfile = np.load(weight.npz)
    #model.load_weights(weight_npzfile['arr_0'])
    print(f"Sucessfully load {checkpoint_load}")
  model.save_weights(checkpoint_path.format(epoch=1))
  print("** checkpoint 001 (init.) saved **")
  history = model.fit(
                  preprocessor.preprocess(server_train_data, rng, train=True, BruteForce_kill_nan=True, add_minmax=False, normal_mode=False),
                  #tfds_train.map(server_train_data, num_parallel_calls=tf.data.AUTOTUNE).batch(HyperSet_Local_batch_size).prefetch(20),
                  epochs=HyperSet_round,
                  verbose=2,
                  validation_data=preprocessor.preprocess(server_test_data, rng, train=False, BruteForce_kill_nan=True, add_minmax=False, normal_mode=False),
                  callbacks=[CustomCallback(), cp_saver], #cp_recovery,
                )
  model.save_weights(checkpoint_path.format(epoch=HyperSet_round))
  print("** checkpoint final saved **")
  # 暫存訓練結果
  Training_result_distributed["loss"] = history.history["loss"]
  Training_result_distributed["accuracy"] = history.history["accuracy"]
  Training_result_distributed["sparse_top_k_categorical_accuracy"] = history.history["sparse_top_k_categorical_accuracy"]
  Testing_result_centralized["loss"] = history.history["val_loss"]
  Testing_result_centralized["accuracy"] = history.history["val_accuracy"]
  Testing_result_centralized["sparse_top_k_categorical_accuracy"] = history.history["val_sparse_top_k_categorical_accuracy"]
    
  # 儲存結果
  if SAVE:
    FL_Results_folder = secure_mkdir("FL_Results"+"/"+model_name)
    if Training_result_distributed is not None:
        np.savez(f"{FL_Results_folder}/Training_result_distributed.npz", Training_result_distributed)
    if Testing_result_centralized is not None:
        np.savez(f"{FL_Results_folder}/Testing_result_centralized.npz", Testing_result_centralized)

    checkpoint_folder = secure_mkdir("ckpoint"+"/"+model_name)
    print(f"****Saving model weights...****")
    GlobalModel_NewestWeight = model.get_weights()
    np.savez(f"{checkpoint_folder}/final-round-weights.npz", *GlobalModel_NewestWeight)
    #model.save_weights(checkpoint_path.format(epoch=epochs))
    #model.save(model_path)

    # 移除緊急暫存
    #os.rmdir(tmpbackup_folder)

# 緊急狀況備份
except KeyboardInterrupt or InterruptedError:
  print("KeyboardInterrupt or InterruptedError!!")
  print("Saving model...")
  GlobalModel_NewestWeight = model.get_weights()
  np.savez(f"{checkpoint_folder}/interrupt-round-weights.npz", *GlobalModel_NewestWeight)
  print("Model saved.")

  print("Saving result...")
  FL_Results_folder = secure_mkdir("FL_Results"+"/"+model_name)
  if Training_result_distributed is not None:
      np.savez(f"{FL_Results_folder}/Training_result_distributed.npz", Training_result_distributed)
  if Testing_result_centralized is not None:
      np.savez(f"{FL_Results_folder}/Testing_result_centralized.npz", Testing_result_centralized)
  print("Result saved.")

  print("Logging...")
  now_time = datetime.now()
  time_str = now_time.strftime("%m_%d_%Y__%H_%M_%S")
  log_folder = secure_mkdir("FL_log"+"/"+"InterruptSaved_"+model_name)
  log_text = f'*** Centralized Traing Record ***\n \
             *[This training was unexpectly interrupted.]*\n \
             *[Interrupt at epoch = {TraceCurrentEpoch}]*\n \
             Model Name: {model_name}\n \
             FL Finish Time: {time_str}\n \
             \n--- FL setting ---\n \
             Aggregation: {Aggregation_name}\n \
             Rounds: {HyperSet_round}\n \
             Traing population: {HyperSet_Train_all_connect_client_number}\n \
             Testing population: {HyperSet_Test_all_connect_client_number}\n \
             Number of client per round (training): {HypHyperSet_Train_EveryRound_client_number}\n \
             Number of client per round (testing): {HypHyperSet_Test_EveryRound_client_number}\n \
             \n--- Server-side hyperparemeter ---\n \
             Learning-rate: {HyperSet_Server_eta}\n \
             Tau: {HyperSet_Server_tau}\n \
             Beta-1: {HyperSet_Server_beta1}\n \
             Beta-2: {HyperSet_Server_beta2}\n \
             \n--- Client-side hyperparemeter ---\n \
             Learning-rate: {HyperSet_Local_eta}\n \
             Momentum: {HyperSet_Local_momentum}\n \
             Local epoch: {HyperSet_Local_epoch}\n \
             Local batch size: {HyperSet_Local_batch_size}\n \
             \n--- Other env. setting ---\n \
             Random Seed: {SEED}\n \
             \n--- Result ---\nCannot save in this mode.'
  mylog(log_text, log_folder+'/log')
  print("Log saved.")
  sys.exit()

if SAVE:
  now_time = datetime.now()
  time_str = now_time.strftime("%m_%d_%Y__%H_%M_%S")
  N = 100 # To calculate the avg N rounds result.
  Train_Loss_avgN, Train_Acc_avgN, Train_TopKAcc_avgN = Result_Avg_Last_N_round(Training_result_distributed,N)
  Test_Loss_avgN, Test_Acc_avgN, Test_TopKAcc_avgN = Result_Avg_Last_N_round(Testing_result_centralized,N)

  log_folder = secure_mkdir("FL_log"+"/"+model_name)
  log_text = f'*** Centralized Traing Record ***\n' \
             f'Model Name: {model_name}\n' \
             f'FL Finish Time: {time_str}\n' \
             f'\n--- FL setting ---\n' \
             f'Aggregation: {Aggregation_name}\n' \
             f'Rounds: {HyperSet_round}\n' \
             f'Traing population: {HyperSet_Train_all_connect_client_number}\n' \
             f'Testing population: {HyperSet_Test_all_connect_client_number}\n' \
             f'Number of client per round (training): {HypHyperSet_Train_EveryRound_client_number}\n' \
             f'Number of client per round (testing): {HypHyperSet_Test_EveryRound_client_number}\n' \
             f'\n--- Server-side hyperparemeter ---\n' \
             f'Learning-rate: {HyperSet_Server_eta}\n' \
             f'Tau: {HyperSet_Server_tau}\n' \
             f'Beta-1: {HyperSet_Server_beta1}\n' \
             f'Beta-2: {HyperSet_Server_beta2}\n' \
             f'\n--- Client-side hyperparemeter ---\n' \
             f'Learning-rate: {HyperSet_Local_eta}\n' \
             f'Momentum: {HyperSet_Local_momentum}\n' \
             f'Local epoch: {HyperSet_Local_epoch}\n' \
             f'Local batch size: {HyperSet_Local_batch_size}\n' \
             f'\n--- Other env. setting ---\n' \
             f'Random Seed: {SEED}\n' \
             f'\n--- Result ---\n' \
             f'--Last result--\n\
             *Last Train Acc.: {Training_result_distributed["accuracy"][-1]}\n \
             Last Train TopK-Acc.: {Training_result_distributed["sparse_top_k_categorical_accuracy"][-1]}\n \
             Last Train Loss: {Training_result_distributed["loss"][-1]}\n \
             *Last Test Acc.: {Testing_result_centralized["accuracy"][-1]}\n \
             Last Test TopK-Acc.: {Testing_result_centralized["sparse_top_k_categorical_accuracy"][-1]}\n \
             Last Test Loss: {Testing_result_centralized["loss"][-1]}\n' \
             f'--Avg last {N} rounds result--\n' \
             f'*Train Acc. (Avg last {N} rounds): {Train_Acc_avgN}\n' \
             f'Train TopK-Acc. (Avg last {N} rounds): {Train_TopKAcc_avgN}\n' \
             f'Train Loss (Avg last {N} rounds): {Train_Loss_avgN}\n' \
             f'*Test Acc. (Avg last {N} rounds): {Test_Acc_avgN}\n' \
             f'Test TopK-Acc. (Avg last {N} rounds): {Test_TopKAcc_avgN}\n' \
             f'Test Loss (Avg last {N} rounds): {Test_Loss_avgN}\n'
  mylog(log_text, log_folder+'/log')
  print("log saved.")

