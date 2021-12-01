import numpy as np

def Result_AvgLast100round(forderpath, filename,):
    results_np = np.load(f"{forderpath}/{filename}", allow_pickle=True)
    '''解壓縮、匯入'''
    #print(results_np.files) # ['arr_0']
    #print(results_np['arr_0']) # {'loss': [], 'accuracy': [], 'top_k_categorical_accuracy': []}
    
    # 出現 0-d array error，解法: https://chunchengwei.github.io/ruan-jian/jie-jue-numpy-0d-arrays-error/
    results_np = np.atleast_1d(results_np['arr_0'])
    results_dict = dict(results_np[0])

    rounds = len(results_dict['loss'])
    if rounds < 100:
      prompt_str = f'Last round '
      print(prompt_str+'loss',np.mean(results_dict['loss'][-1]))
      print(prompt_str+'Acc',np.mean(results_dict['accuracy'][-1]))
      print(prompt_str+'TopK Acc',np.mean(results_dict['sparse_top_k_categorical_accuracy'][-1]))
    else:
      prompt_str = f'Last {100} avg '
      print(prompt_str+'loss',np.mean(results_dict['loss'][-100:]))
      print(prompt_str+'Acc',np.mean(results_dict['accuracy'][-100:]))
      print(prompt_str+'TopK Acc',np.mean(results_dict['sparse_top_k_categorical_accuracy'][-100:]))

def main():
    forderpath = r"D:\KuihaoFL\Implement_google_AdaptiveFL\FL_Results\11_26_2021__22_12_08_cifar100_noniid_resnet18_fedadagrad"
    print("Train:")
    Result_AvgLast100round(forderpath, "Training_result_distributed.npz")
    print("Test:")
    Result_AvgLast100round(forderpath, "Testing_result_distributed.npz")

main()
