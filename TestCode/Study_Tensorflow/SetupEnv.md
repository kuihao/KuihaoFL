# Anaconda 建置特定的 Tensorflow-gpu 版本

1. 檢查 CUDA toolkit 版本
    * 輸入 nvcc --version
        * Cuda compilation tools, **release 11.5**, V11.5.50
        * **release** 後面所述就是版本
    * Ref: 
        * https://varhowto.com/check-cuda-version/#Method_1_%E2%80%94_Use_nvcc_to_check_CUDA_version
2. 於 Anaconda 建置環境
    * 輸入 conda create -n tf-gpu-cuda8 tensorflow-gpu cudatoolkit=9.0
    * Ref: https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/
3. Jyputer Notebook 加入 tf-gpu 環境
    * 
    * Ref: 


(?) ipython kernel install --user --name=tfds-gpu

conda create -n tf115 tensorflow-gpu cudatoolkit=11.5
conda env remove -n 



[jupyter]
首先照著上面做的進入虛擬環境，然後安裝 ipykernel： conda install ipykernel
幫 Jupyter 增加新的 kernel： python -m ipykernel install --name another_name
接著列出可以使用的 kernel，檢查是否成功：jupyter kernelspec list
jupyter kernelspec remove [mykernel]