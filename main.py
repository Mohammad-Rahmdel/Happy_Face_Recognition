import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from keras import regularizers



X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

# print ("number of training examples = " + str(X_train.shape[0]))
# print ("number of test examples = " + str(X_test.shape[0]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))


input_shape = X_train.shape[1]
X_input = Input(shape= (input_shape,input_shape,3))
X = ZeroPadding2D((3, 3))(X_input)

# CONV -> Batch Normalization -> RELU Block applied to X
# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
lambd = 1e-4
X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', kernel_regularizer=regularizers.l2(lambd))(X)  # n_H = n_W = 7  n_C = 32
X = BatchNormalization(axis = 3, name = 'bn0')(X)
X = Activation('relu')(X)

# MAXPOOL
X = MaxPooling2D((2, 2), name='max_pool')(X)

# FLATTEN X (means convert it to a vector) + FULLYCONNECTED
X = Flatten()(X)
X = Dropout(1)(X)
X = Dense(1, activation='sigmoid', name='fc')(X)
X = Dropout(1)(X)

# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
model = Model(inputs = X_input, outputs = X, name='HappyModel')


"""
You have now built a function to describe your model. To train and test this model, there are four steps in Keras:
    1. Create the model by calling the function above
    2. Compile the model by calling model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])
    3. Train the model on train data by calling model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)
    4. Test the model on test data by calling model.evaluate(x = ..., y = ...)
"""

model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(x = X_train, y = Y_train, epochs = 20, batch_size = 32, shuffle=True)
# model.summary()
# # plot_model(model)

preds = model.evaluate(X_train, Y_train)
print ("Test Accuracy = " + str(preds[1]))
preds = model.evaluate(X_test,Y_test)
print ("Test Accuracy = " + str(preds[1]))



""" Result : 
optimizer = "Adam", loss = "binary_crossentropy"
epochs = 15, batch_size = 32, shuffle=True

Test Accuracy = 0.9933333333333333
Test Accuracy = 0.9599999976158142


epochs = 40
Test Accuracy = 0.9916666666666667
Test Accuracy = 0.9733333373069764


L2 lambda = 1e-5
awful!


epochs = 30
L2 lambda = 1e-5
Test Accuracy = 0.9983333333333333
Test Accuracy = 0.9599999976158142

L2 lambda = 3e-6
Test Accuracy = 0.9933333333333333
Test Accuracy = 0.9799999976158142



keep_prop = 0.4
awful

keep_prop = 0.9
awful
"""
