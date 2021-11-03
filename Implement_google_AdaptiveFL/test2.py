import tensorflow as tf
from mymodel.resnet18 import ResNet18

model = ResNet18((32,32,3),10)
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