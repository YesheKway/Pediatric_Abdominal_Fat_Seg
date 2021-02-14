#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:22:02 2020
@author: ymk
"""

from training import  k_fold_training
from evaluation import Evaluater 
import argparse
import yaml

def main():
    print('starting program ...')
    parser = argparse.ArgumentParser(prog='MedSeg')
    parser.add_argument('-config', '--config_path', type=str, default='./config/default_config.yaml', help='Configuration file defining training and evaluation parameters')
    args = parser.parse_args()
    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    print('='*40)
    print('Starting Training...')
    # k_fold_training(config)        
    print('Training completed...')
    print('='*40)
    print('Starting Evaluation...')
    ev = Evaluater()
    ev.EvaluateModels(config)
    print('Evaluation completed')
    print('='*40)
    
if __name__ == '__main__':
    main()
