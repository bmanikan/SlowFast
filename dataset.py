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

class videoDataset(Sequence):
  def __init__(self, datapath, mode='train', clip_len=8, frame_sample_rate=1, batch_size=2):
    self.clip_len = clip_len
    self.mode = mode
    self.batch_size = batch_size

    self.short_side = [128,160]
    self.crop_size = 112
    self.frame_sample_rate = frame_sample_rate
    folder = os.path.join(datapath,mode)

    self.fnames, self.labels = [],[]
    for label in os.listdir(folder):
      for fname in os.listdir(os.path.join(folder,label)):
        self.fnames.append(os.path.join(folder,label,fname))
        self.labels.append(label)
    self.label2index = {label:index for index,label in enumerate(os.listdir(folder))}
    self.n_labels = len(self.label2index.keys())
    self.label_array = [self.label2index[label] for label in self.labels]

    label_file = str(len(os.listdir(folder)))+'class_labels.txt'
    with open(label_file,'w') as f:
      for index,label in enumerate(sorted(self.label2index)):
        f.writelines(str(index+1) + ' ' +label+'\n')
    
    np.random.seed(43)

  def loadVideo(self,fname):
    remainder = np.random.randint(self.frame_sample_rate)
    cap = cv2.VideoCapture(fname)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_height < frame_width:
      resize_height = np.random.randint(self.short_side[0], self.short_side[1]+1)
      resize_width = int(float(resize_height) / frame_height*frame_width)
    else:
      resize_width = np.random.randint(self.short_side[0], self.short_side[1]+1)
      resize_height = int(float(resize_width) / frame_width*frame_height)

    start_idx = 0
    end_idx = frame_count - 1
    frame_count_sample = (frame_count // self.frame_sample_rate) - 1
    if frame_count > 300:
      end_idx = np.random.randint(300,frame_count)
      start_idx = end_idx - 300
      frame_count_sample = (301 // self.frame_sample_rate) - 1
    buffer = np.zeros((frame_count_sample, resize_height, resize_width, 3),np.dtype('float32'))

    count = 0
    ret = True
    sample_count = 0
    while (count <= end_idx) and ret :
      ret,frame = cap.read()
      if count <= start_idx:
        count+=1
        continue
      if ret == False or count>end_idx:
        break
      if sample_count < frame_count_sample:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize_width,resize_height))
        buffer[sample_count] = frame
        sample_count += 1
      count+=1
    cap.release()
    return buffer

  def __getitem__(self, index):
    buffer_array = []
    labels = []
    f_array = self.fnames[index*self.batch_size:(index+1)*self.batch_size]
    for f_name in f_array:
      buffer = self.loadVideo(f_name)
      while buffer.shape[0] < self.clip_len+2:
        index += 1 
        buffer = self.loadVideo(self.fnames[index])

      #buffer = self.randomFlip(buffer)
      buffer = self.crop(buffer, self.clip_len, self.crop_size)
      buffer = self.normalize(buffer)
      buffer_array.append(tf.convert_to_tensor(buffer))
      labels.append(tf.convert_to_tensor(to_categorical(self.label_array[index],num_classes=self.n_labels)))
    return tf.stack(buffer_array,axis=0),tf.stack(labels,axis=0)

  def crop(self, buffer, clip_len, crop_size):
    time_idx = np.random.randint(buffer.shape[0] - clip_len)
    height_idx = np.random.randint(buffer.shape[1] - crop_size)
    width_idx = np.random.randint(buffer.shape[2] - crop_size)
    return buffer[time_idx:time_idx + clip_len,
                  height_idx:height_idx + crop_size,
                  width_idx:width_idx + crop_size,:]
  def normalize(self,buffer):
    for i, frame in enumerate(buffer):
      frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
      buffer[i] = frame
    return buffer

  def randomFlip(self,buffer):
    if np.random.random() < 0.5:
      for idx,frame in enumerate(buffer):
        frame = cv2.flip(frame,flipCode=1)
        buffer[idx] = frame
    return buffer
  
  def __len__(self):
    return len(self.fnames) // self.batch_size

