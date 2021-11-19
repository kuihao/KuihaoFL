'''
matplotlib 教學
(1) https://towardsdatascience.com/the-many-ways-to-call-axes-in-matplotlib-2667a7b06e06
(2) https://towardsdatascience.com/plt-xxx-or-ax-xxx-that-is-the-question-in-matplotlib-8580acf42f44
'''
import matplotlib.pyplot as plt
import numpy as np
#import argparse
from matplotlib.ticker import MaxNLocator

def myplot(forderpath, filename,):
    results_np = np.load(f"{forderpath}/{filename}", allow_pickle=True)
    '''解壓縮、匯入'''
    #print(results_np.files) # ['arr_0']
    #print(results_np['arr_0']) # {'loss': [], 'accuracy': [], 'top_k_categorical_accuracy': []}
    
    # 出現 0-d array error，解法: https://chunchengwei.github.io/ruan-jian/jie-jue-numpy-0d-arrays-error/
    results_np = np.atleast_1d(results_np['arr_0'])
    results_dict = dict(results_np[0]) # y軸數值
    #print(results_dict['loss'])
    x_ticks = []
    if filename == 'Training_result_distributed.npz':
        x_ticks = np.arange(1, len(results_dict['loss'])+1) # x軸數值
        print("hi")
    elif filename == 'Testing_result_centralized.npz':
        x_ticks = np.arange(0, len(results_dict['loss'])) # x軸數值
    
    fig, axs = plt.subplots(2, 1, figsize=(6, 12))
    '''畫布、繪製子圖(axes)，當子圖的維度有1時，axes的矩陣只能用一維[1]，而非2維[0,1]'''
    
    axs[0].plot(x_ticks, results_dict['loss'])
    axs[0].set_title('Loss') # 子圖標題
    axs[0].legend(['loss'], loc='upper left') # 圖示說明
    axs[1].plot(x_ticks, results_dict['accuracy'], 'tab:orange')
    axs[1].plot(x_ticks, results_dict['top_k_categorical_accuracy'], 'tab:green')
    axs[1].legend(['accuracy', 'top_k_categorical_accuracy'], loc='upper left')
    axs[1].set_title('Acc. & Top-5 Acc.')

    for ax in axs.flat:
        ax.set(ylabel='value') # y軸標籤
        ax.tick_params(axis='both', which='both') # which='both' 所有子圖都要顯示刻度值
        #ax.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) #刻度用整數
    axs[1].set(xlabel='rounds') # x軸標籤

    fig.suptitle('FL Result', fontsize=16)

    plt.show()

def fix_input_main():
    forderpath = r"C:\Users\kuiha\OneDrive - 國立成功大學 National Cheng Kung University\NCKU研究所\FL論文andCode\FlowerFL_code\Implement_Flower_resnet18_EC_SavingResult\FL_Results\11_01_2021__12_05_46_test_save_result"
    myplot(forderpath, "Training_result_distributed.npz")
    #myplot(forderpath, "Testing_result_centralized.npz")

if __name__ == "__main__":
    fix_input_main()
