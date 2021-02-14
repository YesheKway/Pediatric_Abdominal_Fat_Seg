# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:41:48 2019

Training Utils script containing Data Generators, Loss functions and GPU 
setting definition

@author: Yeshe Kway
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from ImageProcessing import ImageProcessing
from keras import backend as K
import tensorflow as tf
import tensorflow
import numpy as np
import random
import keras
import csv
import os
import re

'============================ GPU settings ==================================='

def init_GPU_Session():
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True  
    session = tf.Session(config=config)
    K.set_session(session)

def clearSession():
    sess = get_session()
    clear_session()
    sess.close()
    K.clear_session()

'============================ loss functions ================================='

def dice_coef(y_true, y_pred):
    smooth=0.00001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels=4):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:, index])
    return dice

def class_weighted_pixelwise_crossentropy(target, output):
    '''
    the class wights for this function were computed by the 'CalculateClassWeightsForDataSet' function withing the 
    Preprocessing class in the ImageProcessig.py file 
    weights = [bg_weight, ssat_weight, dsat_weight, vat_weight]

    Parameters
    ----------
    target : tensor
    output : tensor
    Returns: loss value
    -------
    '''
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [1, 1.5, 1.8, 1.8]
    return -tf.reduce_sum(target * weights * tf.log(output))

def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5    
    ones = K.ones(K.shape(y_true))
    pred = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    gt = y_true
    g1 = ones-y_true
    num = K.sum(pred*gt, (1,2,3))
    den = num + alpha*K.sum(pred*g1,(1,2,3)) + beta*K.sum(p1*gt,(1,2,3))
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T


"=========================== Read and Write help functions ===================="

def readCSVToList(pathToCSV):
    """
    read csv data (all ids need to be strored in first row)
    """    
    lisT = []
    with open(pathToCSV) as f:
        reader = csv.reader(f, delimiter=',')    
        row = next(reader) # get first row      
        for subID in row:
            subID = subID.strip("['")
            subID = subID.strip("]'")
            lisT.append(subID)
        return lisT 

def writeListToCSV(lisT, csvName, dst):
    """
    this function saves a list into a CSV file and saves it in destination dst
    Args: 
        dst: destination where CSV should be saved to
        csvName: name of CSV file 
        lisT: type list         
    """
    filename = csvName + ".csv"
    with open(os.path.join(dst, filename), 'w') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(lisT)
    print("CSV file saved as: " + csvName)


def get_ID_lists(dataPath):
    '''
    Loads train and validatio ID list when saved from 'dataPath'
    Parameters
    ----------
    dataPath : TYPE
        DESCRIPTION.
    Returns
    -------
    trainIDs : TYPE
        DESCRIPTION.
    valIDs : TYPE
        DESCRIPTION.
    '''
    p_trainIDs = os.path.join(dataPath, 'training_IDs.csv')
    p_valIDs = os.path.join(dataPath, 'validation_IDs.csv')
    trainIDs = readCSVToList(p_trainIDs)
    valIDs = readCSVToList(p_valIDs)    
    return trainIDs, valIDs 


def selectTrainValIDs(path, dst, ratio = 70, seed = 10):
    '''
     this function takes a path to the training data and extracts the all 
     filenames and splits them by defined 'ratio' into train and 
     validation set. 
     Args: 
         :path :absolute path to data (raw data or label data)
         :ratio: spliting ratio (trainSetInPercentage) default 70         
         dst: where list will be saved to 
    '''
    filenames = os.listdir(path)
    nrOfFiles = len(filenames)
    
    random.Random(seed).shuffle(filenames)
    
    # split data into train and validations set 
    split = int((ratio/100) * nrOfFiles)     
    print("-splitting: " + str(nrOfFiles) + " Subjects" " (train:" + 
           str(ratio)+"%" + " val:" + str(100-ratio)+"%)")                
    names_train = filenames[:split]
    names_val = filenames[split:]           
    print("   -->" + str(len(names_train)) +" training subjects" )
    print("   -->" + str(len(names_val)) + " validation subjects")        
    print("-save split infomration in CSV")
    
    # save validation subject ids in CSV
    writeListToCSV(extractIDs(names_train), "training_IDs" ,dst)    
    # save training subject ids in CSV    
    writeListToCSV(extractIDs(names_val), "validation_IDs", dst)


    
def extractIDs(lisT):
    """
    this function takes a list of strings and extracts only the numbers found in 
    the stings (IDs)
    """
    return list(map(lambda x: re.findall(r"\d+", x), lisT))        
             

def sort_dir_aphanumerical(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def main():
    
    test = 10


if __name__ == '__main__':
    main()





