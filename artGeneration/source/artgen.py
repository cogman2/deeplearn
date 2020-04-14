import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint
%matplotlib inline

pp = print.PrettyPrinter(indent=4)
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
pp.pprint(model)

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image);


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.shape

    # Reshape a_C and a_G (≈2 lines)
    #    a_C_unrolled = tf.reshape(a_C, [m, n_H * n_W, n_C ])
    #    a_G_unrolled = tf.reshape(a_G, [m, n_H * n_W, n_C ])
    #    a_C_unrolled = tf.transpose(tf.reshape(a_C, [-1]))
    #    a_G_unrolled = tf.transpose(tf.reshape(a_G, [-1]))

    # compute the cost with tensorflow (≈1 line)
    #    J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled)**2) / (4 * n_H * n_W * n_C)

    a_C_unrolled = tf.reshape(tf.transpose(a_C, [0, 3, 1, 2]), [m, n_C, -1])
    a_G_unrolled = tf.reshape(tf.transpose(a_G, [0, 3, 1, 2]), [m, n_C, -1])

    # compute the cost with tensorflow (≈1 line)
    J_content = tf.reduce_sum(tf.squared_difference(a_C_unrolled, a_G_unrolled)) / (4 * n_H * n_W * n_C)
    ### END CODE HERE ###

    return J_content


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    ### START CODE HERE ### (≈1 line)
    GA = None
    ### END CODE HERE ###

    return GA

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))

