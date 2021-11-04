'''Ref.: https://blog.csdn.net/weixin_35811044/article/details/100865340 (解析HTML即可取得程式碼，無須登入)'''
from tensorflow import keras

def Conv2D_BN(inputs, filter, kernel, padding, stride):
    '''Kuihao: 此處與論文不符，activation function RELU 應置於 BN 之後 (arXiv:1512.03385, P4)'''
    outputs = keras.layers.Conv2D(filters=filter,kernel_size=kernel,padding=padding,strides=stride,activation='relu')(inputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    return outputs

def residual_block(inputs,filter,stride,whether_identity_change=False):
    x = Conv2D_BN(inputs, filter[0], kernel=(1,1), padding='same', stride=stride) 
    x = Conv2D_BN(x, filter[1], kernel=(3,3), padding='same', stride=1)
    x = Conv2D_BN(x, filter[2] ,kernel=(1,1), padding='same', stride=1)

  	# 累加必須保持尺寸一致，控制恆等層是否需要變channel數和壓縮尺寸 (Kuihao翻譯: stride>=2會改變tensor's shape，因此shortcut也需要同步做一次shape處理，最後合併時才不會報錯)
    if whether_identity_change:
        identity = Conv2D_BN(inputs, filter[2], kernel=(1,1), padding='same', stride=stride)
        x = keras.layers.add([x,identity])
        return x # Kuiaho: 此與論文不符，return 之前應再過一次 RELU function
    else:
        x = keras.layers.add([x,inputs])
        return x # Kuihao: 此與論文不符，return 之前應再過一次 RELU function

def ResNet50():
	inputs = keras.Input(shape=(224,224,3))
	x = Conv2D_BN(inputs,64,(7,7),'same',2)
	x = keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')(x) # 經實驗得知 strides=1 能使正確率再提升

	x = residual_block(x,[64,64,256],1,True) # Kuihao: 此處輸入維度已於前一層改變，應設 False
	x = residual_block(x,[64,64,256],1)
	x = residual_block(x,[64,64,256],1)

	x = residual_block(x,[128,128,512],2,True)
	x = residual_block(x,[128,128,512],1)
	x = residual_block(x,[128,128,512],1)
	x = residual_block(x,[128,128,512],1)

	x = residual_block(x,[256,256,1024],2,True)
	x = residual_block(x,[256,256,1024],1)
	x = residual_block(x,[256,256,1024],1)
	x = residual_block(x,[256,256,1024],1)
	x = residual_block(x,[256,256,1024],1)
	x = residual_block(x,[256,256,1024],1)

	x = residual_block(x,[512,512,2048],2,True)
	x = residual_block(x,[512,512,2048],1)
	x = residual_block(x,[512,512,2048],1)

	x = keras.layers.AveragePooling2D(pool_size=(7,7))(x)
	x = keras.layers.Flatten()(x)
	x = keras.layers.Dense(17,activation='softmax')(x)

	model = keras.Model(inputs=inputs,outputs=x)
	model.summary()
	return model

model = ResNet50()