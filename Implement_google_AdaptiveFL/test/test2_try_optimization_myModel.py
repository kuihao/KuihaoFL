import tensorflow as tf
import numpy as np
from mymodel.resnet18 import myResNet, ResNet18_Dan
from mypkg.gpu.limitGPU import setGPU
setGPU(mode=1,gpus=tf.config.list_physical_devices('GPU'))
np.random.seed(2021)
tf.random.set_seed(2021)

model = myResNet().ResNet18(input_shape=(32,32,3),ClassNum=10)
'''
MaxPooling2D: pool size 3,3 (論文設定)
Epoch 20/20
- loss: 0.0570 - accuracy: 0.9802 - top_k_categorical_accuracy: 0.5235 
- val_loss: 1.6514 - val_accuracy: 0.7048 - val_top_k_categorical_accuracy: 0.4289

MaxPooling2D: pool size 2,2 (更改)
- loss: 0.0634 - accuracy: 0.9777 - top_k_categorical_accuracy: 0.5226 
- val_loss: 1.5767 - val_accuracy: 0.7092 - val_top_k_categorical_accuracy: 0.5563

shortcut_shape_transform: PaddingMode='valid' (再更改)
- loss: 0.0572 - accuracy: 0.9804 - top_k_categorical_accuracy: 0.5106 
- val_loss: 1.7539 - val_accuracy: 0.6964 - val_top_k_categorical_accuracy: 0.4377

MaxPooling2D(strides=1) (再更改) 
# 顯然 MaxPool strides=1 比論文的 strides=2 還能提升將近 5% val_acc. 
# 但是需要多花6~7秒/epoch
- loss: 0.0200 - accuracy: 0.9931 - top_k_categorical_accuracy: 0.4978 
- val_loss: 1.5696 - val_accuracy: 0.7413 - val_top_k_categorical_accuracy: 0.4511
(相同設定 跑第二次 理論上要相同?)
- loss: 0.0171 - accuracy: 0.9940 - top_k_categorical_accuracy: 0.5373 
- val_loss: 1.6026 - val_accuracy: 0.7334 - val_top_k_categorical_accuracy: 0.5755

MaxPooling2D: pool size 3,3 (再更改) # 改回論文設定 # 目前最佳設定，超越網路作者 # 定版
- loss: 0.0238 - accuracy: 0.9915 - top_k_categorical_accuracy: 0.5476 
- val_loss: 1.4853 - val_accuracy: 0.7546 - val_top_k_categorical_accuracy: 0.5410
'''
#model = ResNet18_Dan(nums_class=10)
#model.build(input_shape=(None,32,32,3))
'''
pool size 2,2
- loss: 0.0091 - accuracy: 0.9973 - top_k_categorical_accuracy: 0.5010 
- val_loss: 1.5385 - val_accuracy: 0.7465 - val_top_k_categorical_accuracy: 0.5650
'''
#model.summary()
tf.keras.utils.plot_model(model, "resnet18_with_shape.png", show_shapes=True)

optimizer = tf.keras.optimizers.SGD(momentum=0.9)
model.compile(optimizer, 
              tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=["accuracy", 'top_k_categorical_accuracy'])

# Load data and model here to avoid the overhead of doing it in `evaluate` itself
(x_train,y_train) ,(x_test,y_test) = tf.keras.datasets.cifar10.load_data() 

# Data preprocessing
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.squeeze(y_train,axis=1)
y_test = tf.squeeze(y_test,axis=1)

history = model.fit(
            x = x_train,
            y = y_train,
            epochs = 20,
            batch_size = 128,
            validation_data=(x_test, y_test)
        ) #validation_split=0.1