import numpy as np

def mylog(text:str, filepath):
    path = filepath+'.txt'
    with open(path, 'a') as f:
        f.write(text+'\n')

def Result_Avg_Last_N_round(results_dict, N:int):
  rounds = len(results_dict['loss'])
  if rounds < N:
    prompt_str = f'Last round '
    loss = np.mean(results_dict['loss'][-1])
    print(prompt_str+'loss',loss)
    acc = np.mean(results_dict['accuracy'][-1])
    print(prompt_str+'Acc',acc)
    topkacc = np.mean(results_dict['sparse_top_k_categorical_accuracy'][-1])
    print(prompt_str+'TopK Acc',topkacc)
  else:
    prompt_str = f'Last {N} avg '
    loss = np.mean(results_dict['loss'][-N:])
    print(prompt_str+'loss',loss)
    acc = np.mean(results_dict['accuracy'][-N:])
    print(prompt_str+'Acc',acc)
    topkacc = np.mean(results_dict['sparse_top_k_categorical_accuracy'][-N:])
    print(prompt_str+'TopK Acc',topkacc)
  
  return loss,acc,topkacc