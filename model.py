import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
import os
import shutil
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.utils import Sequence,to_categorical
from utils import safedirs
from tensorflow.keras.layers import Conv3D, MaxPool3D, Dense, Dropout, ReLU, BatchNormalization
from tensorflow.keras import Model,Sequential

class Residual(Model):
  expansion = 4
  def __init__(self, inDim, outDim, downsample=None, stride=1,head_conv=1):
    super(Residual,self).__init__()
    
    if head_conv == 1:
      self.conv1 = Conv3D(outDim, kernel_size=1,use_bias=False)
      self.bn1 = BatchNormalization()
    elif head_conv == 3:
      self.conv1 = Conv3D(outDim, kernel_size=(3,1,1), padding='same', use_bias=False)
      self.bn1 = BatchNormalization()
    else:
      raise ValueError("Invalid head convolution value")  

    self.conv2 = Conv3D(outDim,kernel_size=(1,3,3), strides=(1,stride,stride), padding='same', use_bias=False)
    self.bn2 = BatchNormalization()

    self.conv3 = Conv3D(outDim*4, kernel_size=1, use_bias=False)
    self.bn3 = BatchNormalization()

    self.relu = ReLU()
    self.downsample = downsample

  def call(self, inputs):
    residual = inputs

    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.conv3(x)
    x = self.bn3(x)

    if self.downsample:
      residual = self.downsample(inputs)
    x += residual

    x = self.relu(x)
    return x


class SlowFast(tf.keras.Model):
  def __init__(self, block=Residual, layers=[3,4,6,3], class_num=10, dropout=0.5):
    super(SlowFast,self).__init__()
    
    self.fast_inplanes = 8
    self.fast_conv1 = Conv3D(8, kernel_size=(5,7,7), strides=(1,2,2), padding='same', use_bias=False)
    self.fast_bn1 = BatchNormalization()
    self.fast_relu = ReLU()
    self.fast_mpool = MaxPool3D(pool_size=(1,3,3), strides=(1,2,2), padding='same')

    self.fast_res2 = self._fastblock_(block, 8, layers[0], stride=1, head_conv=3)
    self.fast_res3 = self._fastblock_(block, 16, layers[1], stride=2, head_conv=3)
    self.fast_res4 = self._fastblock_(block, 32, layers[2], stride=2, head_conv=3)
    self.fast_res5 = self._fastblock_(block, 64, layers[3], stride=2, head_conv=3)

    self.lateral_p1 = Conv3D(8*2, kernel_size=(5,1,1), strides=(8,1,1), padding='same', use_bias=False)
    self.lateral_res2 = Conv3D(32*2, kernel_size=(5,1,1), strides=(8,1,1), padding='same', use_bias=False)
    self.lateral_res3 = Conv3D(64*2, kernel_size=(5,1,1), strides=(8,1,1), padding='same', use_bias=False)
    self.lateral_res4 = Conv3D(128*2, kernel_size=(5,1,1), strides=(8,1,1), padding='same', use_bias=False)
    
    self.slow_inplanes = 64+64//8*2
    self.slow_conv1 = Conv3D(64, kernel_size=(1,7,7), strides=(1,2,2), padding='same', use_bias=False)
    self.slow_bn1 = BatchNormalization()
    self.slow_relu = ReLU()
    self.slow_mpool = MaxPool3D(pool_size=(1,3,3), strides=(1,2,2), padding='same')

    self.slow_res2 = self._slowblock_(block, 64, layers[0], stride=1, head_conv=1)
    self.slow_res3 = self._slowblock_(block, 128, layers[1], stride=2, head_conv=1)
    self.slow_res4 = self._slowblock_(block, 256, layers[2], stride=2, head_conv=3)
    self.slow_res5 = self._slowblock_(block, 512, layers[3], stride=2, head_conv=3)
    
    self.dp = Dropout(dropout)
    self.fc1 = Dense(class_num, use_bias=False)

  def call(self,x):
    fast,lateral = self.fastpath(x[:,::2,:,:,:])
    slow = self.slowpath(x[:,::16,:,:,:], lateral)
    x = tf.concat([fast,slow],axis=1)
    x = self.dp(x)
    x = self.fc1(x)
    return x

  def fastpath(self,x):
    lateral = []

    x = self.fast_conv1(x)
    x = self.fast_bn1(x)
    x = self.fast_relu(x)
    pool1 = self.fast_mpool(x)

    lateral_pool1 = self.lateral_p1(pool1)
    lateral.append(lateral_pool1)

    res2 = self.fast_res2(pool1)
    lateral_fr2 = self.lateral_res2(res2)
    lateral.append(lateral_fr2)

    res3 = self.fast_res3(res2)
    lateral_fr3 = self.lateral_res3(res3)
    lateral.append(lateral_fr3)

    res4 = self.fast_res4(res3)
    lateral_fr4 = self.lateral_res4(res4)
    lateral.append(lateral_fr4)

    res5 = self.fast_res5(res4)
    x = tfa.layers.AdaptiveAveragePooling3D(1)(res5)
    x = tf.reshape(x,[-1,x.shape[4]])

    return x, lateral

  def slowpath(self, x, lateral):
    x = self.slow_conv1(x)
    x = self.slow_bn1(x)
    x = self.slow_relu(x)
    
    x = self.slow_mpool(x)
    x = tf.concat([x,lateral[0]],axis=4)
    
    x = self.slow_res2(x)
    x = tf.concat([x,lateral[1]],axis=4)
    
    x = self.slow_res3(x)
    x = tf.concat([x,lateral[2]],axis=4)
    
    x = self.slow_res4(x)
    x = tf.concat([x,lateral[3]],axis=4)
    
    x = self.slow_res5(x)
    x = tfa.layers.AdaptiveAveragePooling3D(1)(x)
    x = tf.reshape(x,[-1,x.shape[4]])
    return x

  def _fastblock_(self, block, outDim, blocks, stride=1, head_conv=1):
    douwnsample = None
    if stride != 1 or self.fast_inplanes != outDim * block.expansion:
      downsample = Sequential([
                    Conv3D(
                        outDim * block.expansion,
                        kernel_size = 1,
                        strides = (1,stride,stride),
                        padding='same',
                        use_bias = False),
                    BatchNormalization()])

    layers = []
    layers.append(block(self.fast_inplanes, outDim, downsample, stride, head_conv=head_conv))
    self.fast_inplanes = outDim * block.expansion
    for i in range(1,blocks):
      layers.append(block(self.fast_inplanes, outDim, head_conv=head_conv))
    return Sequential([*layers])

  def _slowblock_(self,block, outDim, blocks, stride=1, head_conv=1):
    downsample=None
    if stride != 1 or self.slow_inplanes != outDim * block.expansion:
      downsample = Sequential([
                    Conv3D(
                        outDim*block.expansion,
                        kernel_size = 1,
                        strides = (1,stride,stride),
                        padding='same',
                        use_bias=False),
                    BatchNormalization()])
    layers = []
    layers.append(block(self.slow_inplanes, outDim, downsample, stride, head_conv=head_conv))
    self.slow_inplanes = outDim * block.expansion
    for i in range(1,blocks):
      layers.append(block(self.slow_inplanes, outDim, head_conv=head_conv))
    
    self.slow_inplanes = outDim * block.expansion + outDim * block.expansion // 8 * 2
    return Sequential([*layers])