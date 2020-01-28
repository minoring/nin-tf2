"""Define Network in Network model

Reference: https://github.com/tflearn/tflearn/blob/master/examples/images/network_in_network.py
"""
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D


def define_nin(input_shape):
  """Define Network in Network model

  Returns:
    tf.keras.Model of NIN
  """
  input_tensor = Input(shape=input_shape)
  x = Conv2D(192, 5, activation='relu')(input_tensor)
  x = Conv2D(160, 1, activation='relu')(x)
  x = Conv2D(96, 1, activation='relu')(x)
  x = MaxPool2D(2, strides=2, padding='same')(x)
  x = Dropout(0.5)(x)

  x = Conv2D(192, 5, activation='relu')(x)
  x = Conv2D(192, 1, activation='relu')(x)
  x = Conv2D(192, 1, activation='relu')(x)
  x = MaxPool2D(2, strides=2, padding='same')(x)
  x = Dropout(0.5)(x)

  x = Conv2D(192, 3, activation='relu')(x)
  x = Conv2D(192, 1, activation='relu')(x)
  x = Conv2D(10, 1, activation='relu')(x)
  output_tensor = GlobalAveragePooling2D(x)

  return tf.keras.Model(input_tensor, output_tensor)
