#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 05:09:04 2021
@author: ymk
"""
import os 
import sys 
import yaml
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from evaluation import Evaluater
from training import splitandsave

def main():
    # load config 
    config_path = '//home/kwaygo/Documents/Abdominal_Adipose_Segmentation/Code_For_Git/tmp/Test_Evaluation_Code/default_config_512.yaml'
    # # read in config file 
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # start evaluation
    ev = Evaluater()
    ev.EvaluateModels(config)    
    
    
if __name__ == '__main__':
    main()

# ============================= Help Functions ================================
# def create_train_val_list():
#     pathToData = '/home/kwaygo/Documents/Abdominal_Adipose_Segmentation/Code_For_Git/tmp/Test_Evaluation_Code/ChildrenSampleVolumes/Train_RawData'
#     dst = '/home/kwaygo/Documents/Abdominal_Adipose_Segmentation/Code_For_Git/tmp/Test_Evaluation_Code'
#     splitandsave(os.listdir(pathToData), 100, dst,'children')
