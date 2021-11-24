* 製作 log 檔顯示訓練日期、參數
* 儲存訓練數據 Training_result_distributed is not none 無效，
裡面已存有資料結構，需考慮其他方式辨別

client test functoni 更改 // preprocess 套用  
實驗參數
---
1. 

2. Server 的 loss acc 聚合計算，嘗試用 tensor 給 GPU 計算

3. multithread GPU memory release:
https://github.com/tensorflow/tensorflow/issues/19731
```python
import multiprocessing

def run_inference_or_training(param1, param2, ...):
    ...

if __name__ == '__main__':
    p = multiprocessing.Process(
        target=run_inference_or_training,
        args=(param1, param2, )
    )
    p.start()
    p.join()  # Add this if you want to wait for the process to finish.
```
4. 匯入原model參數、持續原model訓練: 待完成