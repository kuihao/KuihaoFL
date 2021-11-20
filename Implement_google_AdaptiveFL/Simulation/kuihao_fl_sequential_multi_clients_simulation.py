# -*- coding: utf-8 -*-
"""Kuihao_FL_Sequential_Multi-Clients_Simulation

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/kuihao/2b8b376fd307d83661fcd65679cc99ec/kuihao_fl_sequential_multi-clients_simulation.ipynb

#Prepare Dataset
"""

dataset_path = 'dataset/cifar100_noniid/content/zip/cifar100_noniid'

"""#IMPORT PKG"""

import os
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#import sys
# add path to use my package
#sys.path.append('/Implement_FedAdativate')#/home/sheng/document/Kuihao

import numpy as np

from mypkg import (
    ServerArg, 
    ModelNameGenerator,
    secure_mkdir,
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

import tensorflow as tf
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
#HyperSet_Model = myResNet().ResNet18(model_input_shape,model_class_number)
#CNN_Model(model_input_shape,model_class_number)
#myResNet().ResNet18(model_input_shape,model_class_number)
HyperSet_Aggregation = Weighted_Aggregate
HyperSet_Agg_eta = 1 #pow(10,(1/2)) #1e-3
HyperSet_Agg_tau = None #1e-2
HyperSet_Agg_beta1 = 0.9 
HyperSet_Agg_beta2 = 0.99

HyperSet_local_eta = 1e-1 #1e-2 #1e-1
HyperSet_local_momentum = 0
HyperSet_local_epoch = 1

HyperSet_all_connect_client_number = 500
HyperSet_all_connect_test_client_number = 100
HyperSet_every_round_client_number = 10
HyperSet_round = 4000
HyperSet_batch_size = 20

"""#EXE FL
1. Get Global model parameter
2. Client_i training
3. Aggregation -> Global save new weights
4. Client_i evaluting
5. finish one round and repeat step 1~4 until finsih n round 
"""

'''
Build-Model
'''
tf.keras.backend.clear_session()
model = myResNet().ResNet18(model_input_shape,model_class_number)
optimizer = tf.keras.optimizers.SGD(learning_rate=HyperSet_local_eta, momentum=HyperSet_local_momentum)
#optimizer = tf.keras.optimizers.Adam() #learning_rate=1e-5
model.compile( optimizer, 
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=["accuracy", 'top_k_categorical_accuracy'])

GlobalModel_NewestWeight = model.get_weights()

rng = tf.random.Generator.from_seed(110, alg='philox') # Server?
SampleClientID_np = Simulation_DynamicClientSample( 
                            training_client_number=HyperSet_every_round_client_number, 
                            connect_client_number=HyperSet_all_connect_client_number,
                            rounds=HyperSet_round,
                            seed=SEED
                          )
SampleClientID_test_np = Simulation_DynamicClientSample( 
                              training_client_number=HyperSet_every_round_client_number, 
                              connect_client_number=HyperSet_all_connect_test_client_number,
                              rounds=HyperSet_round,
                              seed=SEED+1
                            )

Clients_ModelWeights_list = [[]]*HyperSet_every_round_client_number
Clients_DataSize_np = np.zeros([HyperSet_every_round_client_number])
Clients_Results_list = [{}]*HyperSet_every_round_client_number
Clients_DataSize_test_np = np.zeros([HyperSet_every_round_client_number])
Clients_Results_test_list = [{}]*HyperSet_every_round_client_number

preprocessor = GoogleAdaptive_tfds_preprocessor(
                          global_seed=SEED, 
                          crop_size=24, 
                          batch_zize=HyperSet_batch_size, 
                          shuffle_buffer=100, 
                          prefetch_buffer=20,              
                        )

# --------
# [Global varables]
# --------
Training_result_distributed = {'loss':[],'accuracy':[],'top_k_categorical_accuracy':[]}
'''Clients Training 的聚合結果'''
Testing_result_distributed = {'loss':[],'accuracy':[],'top_k_categorical_accuracy':[]}
'''Clients Testing 的聚合結果'''
Training_result_centralized = {'loss':[],'accuracy':[],'top_k_categorical_accuracy':[]}
'''[未使用] Server Training 的結果'''
Testing_result_centralized = {'loss':[],'accuracy':[],'top_k_categorical_accuracy':[]}
'''Server Testing 的結果'''

server_test_data = myLoadDS(dataset_path+'/server/test/global_test_all', 'tfds')
#server_test_data = tf.data.experimental.load(dataset_path+'/server/test/global_test_all')

for rnd in range(HyperSet_round):
  print(f"**** Round {rnd+1} ****")
  
  # Client Train
  for client_i in range(HyperSet_every_round_client_number):
    # get client id
    cid = SampleClientID_np[client_i,rnd]
    #print(f"\nClient {client_i} execute cid={cid}")
    
    # load data
    tfds_train = myLoadDS(dataset_path+f'/client/train/client_{cid}_train', 'tfds')
    #tfds_train = tf.data.experimental.load(dataset_path+f'/client/train/client_{cid}_train')
    train_len = 100 #sum(1 for _ in tfds_train)
    
    # Local model train
    model.set_weights(GlobalModel_NewestWeight)
    history = model.fit(
                preprocessor.preprocess(tfds_train, rng, train=True, BruteForce_kill_nan=True, add_minmax=False, normal_mode=False),
                #tfds_train.map(simple_cifar100_preprocessor, num_parallel_calls=tf.data.AUTOTUNE).batch(HyperSet_batch_size).prefetch(20),
                epochs=HyperSet_local_epoch,
                verbose=3,
              )
    results = {
        "loss": history.history["loss"][-1],
        "accuracy": history.history["accuracy"][-1],
        "top_k_categorical_accuracy": history.history["top_k_categorical_accuracy"][-1],
    }
    Clients_ModelWeights_list[client_i] = model.get_weights()
    Clients_DataSize_np[client_i] = train_len
    Clients_Results_list[client_i] = results

  # Aggregation
  ReturnResults = [(weight,size) for weight,size in zip(Clients_ModelWeights_list,Clients_DataSize_np)]
  GlobalModel_NewestWeight = HyperSet_Aggregation(GlobalModel_NewestWeight,ReturnResults,HyperSet_Agg_eta) #fl.common.parameters_to_weights()

  # Aggregate clients' training results (loss, acc., top-k-acc.)
  all_dataset_size = Clients_DataSize_np.sum()

  losses_weighted = [r['loss'] * size for r,size in zip(Clients_Results_list,Clients_DataSize_np)]
  loss_aggregated = sum(losses_weighted) / all_dataset_size

  accuracies_weighted = [r['accuracy'] * size for r,size in zip(Clients_Results_list,Clients_DataSize_np)]
  accuracy_aggregated = sum(accuracies_weighted) / all_dataset_size

  topK_accuracies_weighted = [r['top_k_categorical_accuracy'] * size for r,size in zip(Clients_Results_list,Clients_DataSize_np)]
  topK_accuracies_aggregated = sum(topK_accuracies_weighted) / all_dataset_size
  
  # 輸出儲存 Client-side 訓練結果
  print(f"****\nRound {rnd+1}, train results from clients after aggregation:\n"\
  f"Loss:{loss_aggregated} Acc.:{accuracy_aggregated} TopK Acc.:{topK_accuracies_aggregated}"\
  f"\n****")
  Training_result_distributed["loss"].append(loss_aggregated)
  Training_result_distributed["accuracy"].append(accuracy_aggregated)
  Training_result_distributed["top_k_categorical_accuracy"].append(topK_accuracies_aggregated)

  '''
  # 輸出儲存 Server-side 測試結果
  print("**** Server-side evaluate:")
  model.set_weights(GlobalModel_NewestWeight)
  loss_aggregated, accuracy_aggregated, topK_accuracies_aggregated = model.evaluate( 
          preprocessor.preprocess(server_test_data, rng, train=False, add_minmax=True),
          #server_test_data.map(simple_cifar100_preprocessor, num_parallel_calls=tf.data.AUTOTUNE).batch(HyperSet_batch_size).prefetch(20),
          verbose=2,
         )
  print("***")
  Testing_result_centralized["loss"].append(loss_aggregated)
  Testing_result_centralized["accuracy"].append(accuracy_aggregated)
  Testing_result_centralized["top_k_categorical_accuracy"].append(topK_accuracies_aggregated)
  '''

  
  # Client Test
  for client_i in range(HyperSet_every_round_client_number):
    # get client id
    cid = SampleClientID_test_np[client_i,rnd]
    #print(f"\nClient {client_i} test cid={cid}")
    
    # load data
    tfds_test = myLoadDS(dataset_path+f'/client/test/client_{cid%100}_train', 'tfds')
    #tfds_test = tf.data.experimental.load(dataset_path+f'/client/test/client_{cid%100}_train')
    val_len = 100 #sum(1 for _ in tfds_test)
    
    # Local model train
    model.set_weights(GlobalModel_NewestWeight)
    loss, acc, topk = model.evaluate( 
                      preprocessor.preprocess(tfds_test, rng, train=False, BruteForce_kill_nan=True, add_minmax=False, normal_mode=False),
                      #tfds_test.map(simple_cifar100_preprocessor, num_parallel_calls=tf.data.AUTOTUNE).batch(HyperSet_batch_size).prefetch(20),
                      verbose=3,
                      )
    results = {
        "loss": loss,
        "accuracy": acc,
        "top_k_categorical_accuracy": topk,
    }
    Clients_DataSize_test_np[client_i] = val_len
    Clients_Results_test_list[client_i] = results
  
  # Aggregate clients' testing results (loss, acc., top-k-acc.)
  all_dataset_size = Clients_DataSize_test_np.sum()

  losses_weighted = [r['loss'] * size for r,size in zip(Clients_Results_test_list,Clients_DataSize_test_np)]
  loss_aggregated = sum(losses_weighted) / all_dataset_size

  accuracies_weighted = [r['accuracy'] * size for r,size in zip(Clients_Results_test_list,Clients_DataSize_test_np)]
  accuracy_aggregated = sum(accuracies_weighted) / all_dataset_size

  topK_accuracies_weighted = [r['top_k_categorical_accuracy'] * size for r,size in zip(Clients_Results_test_list,Clients_DataSize_test_np)]
  topK_accuracies_aggregated = sum(topK_accuracies_weighted) / all_dataset_size
  
  # 輸出儲存 Client-side 測試結果
  print(f"****\nRound {rnd+1}, test results from clients after aggregation:\n"\
  f"Loss:{loss_aggregated} Acc.:{accuracy_aggregated} TopK Acc.:{topK_accuracies_aggregated}"\
  f"\n****")
  Testing_result_distributed["loss"].append(loss_aggregated)
  Testing_result_distributed["accuracy"].append(accuracy_aggregated)
  Testing_result_distributed["top_k_categorical_accuracy"].append(topK_accuracies_aggregated)
  
  # 定期備份
  if SAVE:
    FL_Results_folder = secure_mkdir("FL_Results"+"/"+model_name)
    if rnd%1000==0:
      if Training_result_distributed is not None:
          np.savez(f"{FL_Results_folder}/Training_result_distributed.npz", Training_result_distributed)
      if Testing_result_distributed is not None:
          np.savez(f"{FL_Results_folder}/Testing_result_distributed.npz", Testing_result_distributed)
      if Testing_result_centralized is not None:
          np.savez(f"{FL_Results_folder}/Testing_result_centralized.npz", Testing_result_centralized)
    elif rnd+1 == HyperSet_round:
      if Training_result_distributed is not None:
          np.savez(f"{FL_Results_folder}/Training_result_distributed.npz", Training_result_distributed)
      if Testing_result_distributed is not None:
          np.savez(f"{FL_Results_folder}/Testing_result_distributed.npz", Testing_result_distributed)
      if Testing_result_centralized is not None:
          np.savez(f"{FL_Results_folder}/Testing_result_centralized.npz", Testing_result_centralized)

  if SAVE:
    checkpoint_folder = secure_mkdir("ckpoint"+"/"+model_name)
    if rnd%1000==0:
      print(f"****Saving round {rnd+1} aggregated_weights...****")
      np.savez(f"{checkpoint_folder}/round-{rnd+1}-weights.npz", *GlobalModel_NewestWeight)
    elif rnd+1 == HyperSet_round:
      print(f"****Saving round {rnd+1} aggregated_weights...****")
      np.savez(f"{checkpoint_folder}/round-{rnd+1}-weights.npz", *GlobalModel_NewestWeight)

  print('\n')
