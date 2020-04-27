# import tensorflow as tf
# from tensorflow import keras
# from tf.keras.models import Sequential
# from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
# from keras.models import Model
# #from keras.layers.normalization import BatchNormalization
# #import keras.layers.normalization.BatchNormalizatin as BatchNormalization
# from keras.layers.pooling import MaxPooling2D, AveragePooling2D
# from keras.layers.merge import Concatenate
# from keras.layers.core import Lambda, Flatten, Dense
# from keras.initializers import glorot_uniform
# from keras.engine.topology import Layer
from tensorflow.keras import backend as K

from tensorflow.keras.layers import *


from faceRecog.inception_blocks import faceRecoModel

K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from faceRecog.fr_utils import *
from faceRecog.inception_blocks import *
from matplotlib import pyplot as plt
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2


if __name__== "__main__":
    np.set_printoptions(threshold=np.nan)

    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    print("Total Params:", FRmodel.count_params())