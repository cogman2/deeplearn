import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from faceRecog.triplet_loss import *

#with tf.Session() as test:
with tf.compat.v1.Session() as test:
    tf.compat.v1.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.compat.v1.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.compat.v1.random_normal([3, 128], mean=1, stddev=1, seed=1),
              tf.compat.v1.random_normal([3, 128], mean=3, stddev=4, seed=1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))