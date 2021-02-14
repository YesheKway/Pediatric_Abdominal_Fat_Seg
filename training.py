#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:12:03 2020

@author: Yeshe Kway 
"""
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Utils import clearSession, init_GPU_Session, dice_coef_multilabel 
from Utils import tversky_loss, class_weighted_pixelwise_crossentropy
from Utils import readCSVToList, writeListToCSV, selectTrainValIDs
from Utils import sort_dir_aphanumerical
from keras.optimizers import Adam
from models import Unet_Designer
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import random
import json
import sys
import os

global ID 
ID = 1


def trainModel(config, dataPath, dst):    
    '''
    Training function: in this function all hyperparameters are defined and then 
    Neural Network is trained 
    '''    
    # Path to train validations IDs 
    p_trainIDs = os.path.join(dataPath, 'training_IDs.csv')
    p_valIDs = os.path.join(dataPath, 'validation_IDs.csv')
    
    # define model    
    um = Unet_Designer()
    model = um.get_model(config)
    # initialize data generators
    trainIDs = readCSVToList(p_trainIDs)
    valIDs = readCSVToList(p_valIDs)
    TrainGen = init_DataGen(config, dataPath, trainIDs)
    ValGen = init_DataGen(config, dataPath, valIDs)
    # testDataGenOutput(TrainGen)
    
    # define/get loss function
    loss_function = get_loss(config['loss_function'])
    if loss_function == 0:
        print('LOSS function defined in config file not found\n: please define loss function')
        print('Program terminated')
        sys.exit()
    
    # compile model and start training
    model.compile(optimizer=Adam(lr=float(config['learning_rate'])), loss=loss_function,  metrics=['accuracy'])   
    print("model is compiled")
    # define CallBacks 
    filepath= os.path.join(dst, "VGG16-cross-lr5-improvement-{epoch:02d}-{val_loss:.2f}.h5")
    es = EarlyStopping(monitor='val_loss', patience= int(config['patience']), verbose=1)
    mc = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1, mode='min')
    CALLBACKS = [es, mc]         
    print("callbacks are defined")       
    print("="*30)    
    print("starting training")    
    history = model.fit_generator(generator = TrainGen, 
                                  validation_data=ValGen, 
                                  epochs=int(config['n_epochs']),
                                  verbose = 1,
                                  callbacks=CALLBACKS)     
    # save training history       
    with open(os.path.join(dst, 'NM-tverl-lr5-History.json'), 'w') as f:
        json.dump(history.history, f)    
    print("training is completed")
    print("="*30)
    print("="*30)   
    
    
def get_loss(name = 'dice_coef_multilabel'):
    if name == 'dice_coef_multilabel':
        return dice_coef_multilabel
    elif name == 'class_weighted_pixelwise_crossentropy':
        return class_weighted_pixelwise_crossentropy
    elif name == 'tversky':
        return tversky_loss
    else:
        return 0
    
    
def init_DataGen(config, dataPath, IDs):
    # define DataGenerator parameters                        
    params_trainGen = {'path_ToTrainData' : dataPath, 
                       'idList': IDs,
                       'batch_size': config['batch_size'],
                       'imagedim': (config['img_dim'][0],config['img_dim'][1]),          
                       'n_InputChannels': config['n_input_channels'],
                       'n_classes': config['n_classes'],
                       'shuffle': config['shuffle'],                       
                       'augmentation': config['augmentation'],
                       'biasCorrection': config['biasfield_correction']}    
    return DataGenerator_AbdominalFatSeg(**params_trainGen)
    
    
def saveSplitInfo(trainIDs, valIDs, name, dst):
    '''
    save train and split info to csv
    '''            
    trainname = os.path.join(name + '_trainIDs')
    valname = os.path.join(name + '_validationIDs')
    writeListToCSV(trainIDs, trainname, dst)
    writeListToCSV(valIDs, valname, dst)    


def rmv_b8c(string):
    '''
    removes 8 last characters from string
    '''
    return string[:-8]


def splitList(list_, ratio, seed =10):
    '''
    split list by ratio (trainratio)
    Args:
        :ratio defines the ratio for amount of training data 
    '''
    nrOfFiles = len(list_)
    random.Random(seed).shuffle(list_)
    # split data into train and validations set 
    split = int((ratio/100) * nrOfFiles)     
    print("-splitting: " + str(nrOfFiles) + " Subjects" " (train:" + 
          str(ratio)+"%" + " val:" + str(100-ratio)+"%)")                
    names_train = list_[:split]
    names_val = list_[split:]           
    print("   -->" + str(len(names_train)) +" training subjects" )
    print("   -->" + str(len(names_val)) + " validation subjects")        
    print("-save split infomration in CSV")         
    
    names_train = [rmv_b8c(item) for item in names_train]
    names_val = [rmv_b8c(item) for item in names_val]
    
    return names_train, names_val         
    

def splitandsave(list_, ratio, dst, name, seed =10):
    '''
    splits and save lists 
    '''
    trainlist , vallist = splitList(list_, ratio, seed =seed)
    saveSplitInfo(trainlist , vallist, name, dst)   
    
    
def loadRawAndLabel(pathtoData, subjectName):
    '''
    loading raw and corresponding label volume
    '''
    sub_label = subjectName + '_label.npy'
    raw_label = subjectName + '_raw.npy'
    # load label 
    pathTolabel = os.path.join(pathtoData, 'Train_LabelData')
    abs_label = os.path.join(pathTolabel, sub_label)
    label_volume = np.load(abs_label)
    # load raw     
    pathToraw = os.path.join(pathtoData, 'Train_RawData')
    abs_raw = os.path.join(pathToraw, raw_label) 
    raw_volume = np.load(abs_raw)
    return raw_volume, label_volume
    

def extractImagesFromVolumes(ID_list, pathTodata, dst_train_Rawdir ,dst_train_Labeldir):
    ''' 
    this function extracts all images from volumes listed in ID_list and 
    stores them to dst
    '''    
    global ID
    for subject in tqdm(ID_list):     
        print(subject)                   
        # extract ROI
        raw_images, label_images = loadRawAndLabel(pathTodata, subject)                                    
#        print('-'*30)
        print('extracting image from', subject)            
#        # save raw and label data in seperate folders with IDs
        for n in range(len(raw_images)):                                    
            image_raw_name = str(ID) + '_raw'
            image_label_name = str(ID) + '_label'                
            # create paths to save arrays to  
            raw_path = os.path.join(dst_train_Rawdir , image_raw_name)
            label_path = os.path.join(dst_train_Labeldir, image_label_name)                                                
            # save arrays
            np.save(raw_path, raw_images[n])
            np.save(label_path, label_images[n]) 
#            print(ID)
            ID += 1        
        print( n+1, 'slices extracted' )                    
    print('-'*30)
    print('Finish')
    print('-'*30) 
    print('total slices extracted :', ID-1)      
    
    
def ExtractTrainingData(childrenDataPath, neonatalDataPath, pathToChildrenTrainList, pathToNeonatalTrainList, dst):        
    '''
    This method images from all volumes noted in ID_list from subjects volumes 
    stored at dataPath to TargetPath.
    '''       
    global ID 
    ID = 1    
    print('-'*30)
    print('Extracting training data...')
    print('-'*30)                   
    # create main  
    main_dir = 'TrainData_' + "_autocreated"
    main_dir =  os.path.join(dst, main_dir)    
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)            
    # create directory for train folder 
    train_Rawdir = os.path.join(main_dir, 'Train_RawData')
    if not os.path.exists( train_Rawdir):
        os.mkdir( train_Rawdir)            
    # create directory for validation folder
    train_Labeldir =  os.path.join(main_dir, 'Train_LabelData')
    if not os.path.exists(train_Labeldir):
        os.mkdir(train_Labeldir)
    print('Folders created')
    
    childrenIDs = readCSVToList(pathToChildrenTrainList)
    neonatalIDs = readCSVToList(pathToNeonatalTrainList)
    # extract children images 
    extractImagesFromVolumes(childrenIDs, childrenDataPath, train_Rawdir , train_Labeldir)
    # exc neonatal images 
    extractImagesFromVolumes(neonatalIDs, neonatalDataPath, train_Rawdir , train_Labeldir)


    
def CreateTrainValListsForTrainig(pathToExtractedTrainingData):
    '''
    creaste a nother train val split for monitoring training process
    '''
    pathToFiles = os.path.join(pathToExtractedTrainingData, 'Train_RawData')
    ratio = 70
    selectTrainValIDs(pathToFiles, pathToExtractedTrainingData, ratio)    
    
    
    
def k_fold_training(config):    
    
    trainedModels_dir = os.path.join(config['dst'], 'models')
    os.mkdir(trainedModels_dir)
    list_ChildrenData = os.listdir(os.path.join(config['children_data_path'], 
                                                'Train_RawData' ))
    list_NeonatalData = os.listdir(os.path.join(config['infant_data_path'], 
                                                'Train_RawData'))
    for n in range(1, config['experimentalrepeat']+1):
        # create new folder 
        modelpath = os.path.join(trainedModels_dir, 'model_' + str(n))
        os.mkdir(modelpath)
        # create data split for children and neonatal data and save info in .csv
        splitandsave(list_ChildrenData,config['split_ratio_train'], modelpath, 'children', n)
        splitandsave(list_NeonatalData, config['split_ratio_train'], modelpath, 'neonatal', n)
        # define path to train lists 
        children_trainIDs = os.path.join(modelpath, 'children_trainIDs.csv')
        neonatal_trainIDs = os.path.join(modelpath, 'neonatal_trainIDs.csv')
        # extract training data 
        ExtractTrainingData(config['children_data_path'], config['infant_data_path'],
                            children_trainIDs, neonatal_trainIDs, modelpath)        
        pathToExtractedTrainingData = os.path.join(modelpath, 
                                                   'TrainData__autocreated')
        # create train validation lists for training process
        CreateTrainValListsForTrainig(pathToExtractedTrainingData)
        # train on current data split 
        init_GPU_Session()      
        trainModel(config, pathToExtractedTrainingData, modelpath)
        # delete extracted training data 
        shutil.rmtree(pathToExtractedTrainingData )
        print('folder deleted')
        clearSession()


def readExcel(pathtoExcel):
    p_children_excel = os.path.join(pathtoExcel, 'SegResults_Children.xlsx')             
    children_df = pd.read_excel(p_children_excel, index_col=0)    
    p_neonates_excel = os.path.join(pathtoExcel, 'SegResults_Neonates.xlsx') 
    neonatal_df = pd.read_excel(p_neonates_excel, index_col=0)        
    return neonatal_df, children_df

def addToDf(combined_df, df, modelname):
    # dsat info 
    combined_df.loc['DSAT_mean_dice', modelname] = df.loc['dsc_mean','DSAT'] 
    combined_df.loc['DSAT_mean_fp', modelname] = df.loc['fp_mean','DSAT']
    combined_df.loc['DSAT_mean_fn', modelname] = df.loc['fn_mean','DSAT']    
    # ssat info 
    combined_df.loc['SSAT_mean_dice', modelname] = df.loc['dsc_mean','SSAT'] 
    combined_df.loc['SSAT_mean_fp', modelname] = df.loc['fp_mean','SSAT']
    combined_df.loc['SSAT_mean_fn', modelname] = df.loc['fn_mean','SSAT']
    # vat info 
    combined_df.loc['VAT_mean_dice', modelname] = df.loc['dsc_mean','VAT'] 
    combined_df.loc['VAT_mean_fp', modelname] = df.loc['fp_mean','VAT']
    combined_df.loc['VAT_mean_fn', modelname] = df.loc['fn_mean','VAT']            
    return combined_df


def printResults(results_df):
    mean_ = results_df.mean(axis=1)
    STD_ = results_df.std(axis=1)
    max_ = results_df.max(axis=1)
    min_ = results_df.min(axis=1)
    print('='*30)
    print('MINIMUMS_VALUES')    
    print('_'*30)
    print(min_)
    print('='*30)
    print('='*30)
    print('MAXIMUMS_VALUES')    
    print('_'*30)
    print(max_)
    print('='*30)
    print('MEAN_VALUES')    
    print('_'*30)
    print(mean_)
    print('='*30)    
    print('STD_VALUES')    
    print('_'*30)
    print(STD_)
    print('='*30)       
    
    
def computeAverageResults(pathToFolder):        
    '''
    pathToFolders: path to folder with models
    '''
    pathToFolder = str(Path('/home/kwaygo/Documents/Abdominal_Adipose_Segmentation/TrainedModels/MultipleTrainingResults'))
    folders = os.listdir(pathToFolder)
    folders = sort_dir_aphanumerical(folders)
    index = ('DSAT_mean_dice', 'DSAT_mean_fp', 'DSAT_mean_fn','SSAT_mean_dice', 
             'SSAT_mean_fp', 'SSAT_mean_fn', 'VAT_mean_dice', 'VAT_mean_fp', 'VAT_mean_fn') 
    results_df_children = pd.DataFrame(index=index, columns=folders)
    results_df_neonates = pd.DataFrame(index=index, columns=folders)
    
    for modelfolder in tqdm(folders):
        neonatal_df, children_df = readExcel(os.path.join(pathToFolder, modelfolder))
        results_df_children = addToDf(results_df_children, children_df, modelfolder)
        results_df_neonates = addToDf(results_df_neonates, neonatal_df, modelfolder)

    printResults(results_df_children)
    print('#'*50)
    print('NEONATAL RESULTS')
    print('#'*50)
    printResults(results_df_neonates)


# def main():
    # k_fold_training()
#     # startMultipleModelEvaluation()
#     # computeAverageResults('0')
    
# if __name__ == "__main__":
#     main()
