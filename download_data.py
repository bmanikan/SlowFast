
from pathlib import Path
from collections import defaultdict
import urllib.request as urlreq
import numpy as np
import time
import os
import shutil
import glob
from utils import safedirs


def download_data():
  root = os.getcwd()

  urlreq.urlretrieve('http://storage.googleapis.com/thumos14_files/UCF101_videos.zip',root + '/data.zip')
  urlreq.urlretrieve('http://crcv.ucf.edu/THUMOS14/Class%20Index.txt','labels.txt')

  labels = {int(i[0]):i[1] for i in np.loadtxt('/content/labels.txt',dtype=str)}
  filename = root + '/data.zip'
  zip_ref = ZipFile(filename)
  zip_ref.extractall('/content/dataset')
  zip_ref.close()

  dataset = root + '/dataset/UCF101'

  train_dir = os.path.join(dataset , 'train')
  safedirs(train_dir)
  val_dir = os.path.join(dataset , 'validation')
  safedirs(val_dir)
  for i in range(1,len(labels)+1):
    files = glob.glob(dataset + f'/*{labels[i]}*')
    train,test = train_test_split(files,test_size=0.2,shuffle=True)
    safedirs(os.path.join(train_dir, labels[i]))
    safedirs(os.path.join(val_dir, labels[i]))
    for f_train,f_test in zip(train,test):
      try:
        shutil.copy(f_train,os.path.join(train_dir, labels[i]))
        shutil.copy(f_test,os.path.join(val_dir, labels[i]))
      except e:
        raise Exception(e)

if __name__ = '__main__':
  download_data()