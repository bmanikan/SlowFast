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
from dataset import *
from download_data import *
from model import *



download_data()

datapath = '/content/dataset/UCF101'
train_dataset = videoDataset(datapath, mode='train', clip_len=64, batch_size=4)
test_dataset = videoDataset(datapath, mode='validation', clip_len=64, batch_size=4)

#stop it early
earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 20,
    restore_best_weights=True
)

#model checkpoint
checkpointfile = os.path.join(os.getcwd(), 'drive/My Drive/project/plantpathology/ResNet50_09_08/0.95')

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpointfile,
    monitor = 'val_loss',
    save_best_only = True
)
  
#Learning rate scheduler
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.1,
    patience = 7,
    cooldown = 1,
    min_lr = 0.0000001,
    verbose=1
)


model = SlowFast(class_num=101)
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-2,weight_decay=1e-5), 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit_generator(generator=train_dataset,
                              validation_data=test_dataset,
                              epochs=20,
                              verbose=1)
