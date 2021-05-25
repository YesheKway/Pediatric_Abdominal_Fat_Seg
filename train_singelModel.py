#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 13:45:21 2021
@author: ymk
"""
from Utils import init_GPU_Session, get_ID_lists, dice_coef_multilabel
from keras.callbacks import EarlyStopping, ModelCheckpoint
from DataGenerators import DataGenerator_AbdominalFatSeg
from keras.optimizers import Adam
from models import Unet_Designer
from shutil import copyfile
import argparse
import json
import yaml
import sys
import os 

# =============================================================================
#                         Help Functions
# =============================================================================

def process_config_args(config):
    '''
    convert variables from config file file to int of float
    '''
    config["n_output_channels"]  = int(config["n_output_channels"])
    config["n_input_channels"]  = int(config["n_input_channels"])
    config["batch_size"]  = int(config["batch_size"])
    config["dropout"]  = float(config["dropout"])
    config["learning_rate"]  = float(config["learning_rate"])
    config["patience"]  = int(config["patience"])
    config["n_epochs"]  = int(config["n_epochs"])            
    return config


def init_DataGen(config, ID_list):
    # define DataGenerator parameters                        
    params_trainGen = {'path_ToTrainData' : config['data_path'],
                       'idList': ID_list,
                       'batch_size': config["batch_size"],
                       'imagedim': (config['img_dim'][0],config['img_dim'][1]),          
                       'n_InputChannels': config['n_input_channels'],
                       'n_classes': config["n_output_channels"],
                       'shuffle': config["shuffle"],
                       'augmentation': config["augmentation"]}
    return DataGenerator_AbdominalFatSeg(**params_trainGen)
        
        
def startTraining(config):   
    '''
    Training function: creates data generators and executres the training 
    '''
    # create folder at dst
    folder_name = define_foldername(config)
    dst_folder = os.path.join(config['dst'], folder_name)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder) 
    else: sys.exit('folder with this config already exists at dst')        
    
    # copy config file to folder path 
    copyfile(config['config_path'],
             os.path.join(dst_folder,
                          os.path.basename(config['config_path'])))    
    # GPU settings
    init_GPU_Session()
    # load train and validation id lists
    trainIDs, valIDs = get_ID_lists(config['data_path'])
    # create data generators
    TrainGen = init_DataGen(config, trainIDs)
    ValGen = init_DataGen(config, valIDs)    
    # define and compile model
    UD = Unet_Designer()
    model = UD.get_model(config)
    # model.summary()
    model.compile(optimizer=Adam(lr=config['learning_rate']), loss=get_loss(config['loss_function']),  metrics=['accuracy'])   
    print("model is compiled")
    # define CallBacks 
    filepath = os.path.join(dst_folder, "model-improvement-{epoch:02d}-{val_loss:.2f}.h5")
    es = EarlyStopping(monitor='val_loss', patience= config['patience'], verbose=1)
    mc = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1, mode='min')
    CALLBACKS = [es, mc]         
    print("callbacks are defined")       
    print("="*30)    
    print("starting training")    
    # start training
    history = model.fit_generator(generator = TrainGen, validation_data = ValGen , epochs= config["n_epochs"], verbose = 1, callbacks=CALLBACKS)     
    # save training history       
    with open(os.path.join(dst_folder, 'Trainig-History.json'), 'w') as f:
        json.dump(history.history, f)    
    print("training is completed")
    print("="*30)
    print("="*30)   

def get_loss(loss_function):
    if loss_function == 'dice_coef_multilabel':
        return dice_coef_multilabel

def define_foldername(config):
    folder_name = config['model_type'] + '_bz' + str(config['batch_size']) + '_utilize_pretrained_weights' + str(config['utilize_pretrained_weights']) + '_n_input_channels' + str(config['n_input_channels'])
    return folder_name 


def main():
    print('starting program ...')
    parser = argparse.ArgumentParser(prog='TrainParameters')
    parser.add_argument('-config', '--config_path', type=str, default='./config/singleTraining_config.yaml', help='Configuration file defining training and evaluation parameters')
    args = parser.parse_args()
    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config['config_path'] = args.config_path
    config = process_config_args(config)

    print('='*40)
    print('Starting Training...')
    startTraining(config)
    print('Training completed...')
    
    print('='*40)
    
if __name__ == '__main__':
    main()

