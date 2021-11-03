import tensorflow as tf
from mymodel.resnet18 import ResNet18

model = ResNet18((32,32,3),10)
#model.summary()
tf.keras.utils.plot_model(model, "resnet18_with_shape.png", show_shapes=True)