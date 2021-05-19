import os

def safedirs(path):
  if not os.path.exists(path):
    os.makedirs(path)