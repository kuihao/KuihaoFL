from tensorflow import keras
'''https://blog.csdn.net/weixin_35811044/article/details/100865340'''

def Conv2D_BN(inputs, filter, kernel, padding, stride):
    '''We adopt batch normalization (BN) right after each convolution (arXiv:1512.03385, P4)'''
    outputs = keras.layers.Conv2D(filters=filter,kernel_size=kernel,padding=padding,strides=stride,activation='relu')(inputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    return outputs

def residual_block(inputs,filter,stride,whether_identity_change=False):
    x = Conv2D_BN(inputs, filter[0], kernel=(1,1), padding='same', stride=stride) 
    x = Conv2D_BN(x, filter[1], kernel=(3,3), padding='same', stride=1)
    x = Conv2D_BN(x, filter[2] ,kernel=(1,1), padding='same', stride=1)

  # 累加必须保持尺寸一致，控制恒等层是否需要变channel数和压缩尺寸
    if whether_identity_change:
        identity = Conv2D_BN(inputs, filter[2], kernel=(1,1), padding='same', stride=stride)
        x = keras.layers.add([x,identity])
        return x
    else:
        x = keras.layers.add([x,inputs])
        return x

def ResNet():
	inputs = keras.Input(shape=(224,224,3))
	x = Conv2D_BN(inputs,64,(7,7),'same',2)
	x = keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')(x)

	x = residual_block(x,[64,64,256],1,True)
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
