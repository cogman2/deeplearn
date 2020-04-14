#import tensorflow as tf

#from tensorflow.python.keras import *
import tensorflow as tf
import numpy as np

#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Conv2D
#from tensorflow.python.keras.layers import MaxPooling2D
#from tensorflow.python.keras.layers import Flatten
#from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import * #Dense Flatten MaxPooling2D Conv2D

# import tensorflow.keras.layers.MaxPooling2D as MaxPooling2D
# import tensorflow.keras.layers.Conv2D as Conv2D
from tensorflow.python.keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator

# from tensorflow.python.keras.preprocessing import image
from keras_preprocessing import image
ROOT ='/home/robertsneddon/projects/dl'

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# The data fitting portion of the code


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(ROOT+'/dataset/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(ROOT+'/dataset/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 2000)

test_image = image.load_img(ROOT+'/dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)

training_set.class_indices

result

