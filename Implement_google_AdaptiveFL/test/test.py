import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
counter = tf.data.experimental.Counter()
for i in range(4):
    print(counter)