# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:03:27 2019

@author: ymk
"""
from pathlib import Path
import keras
import tensorflow as tf 
import numpy as np
import os
import random


def main():

    print('tensorflow successfully loaded')
    print(tf.test.is_gpu_available())

if __name__ == "__main__":
      main()
