#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 14:20:14 2021
@author: ymk
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ImageProcessing import ImageProcessing
from Utils import readCSVToList
import numpy as np
import random
import keras 
import re
import os 

'============================  DataGenerators  ================================'

class DataGenerator_AbdominalFatSeg(keras.utils.Sequence):
    '''
    this generator load numpy image files to provide them to the Neural Network.
    resitzing infant data by zerro padding 
    
    Args:
        path_ToTrainData : path to the image data                 
        idList           : list containing the ids for training/validation        
        batch_size       : Batch size 
        imagedim         : image dimention shape = (w, h)
        n_InputChannels  : number of input channels 
        n_classes        : number of classes
        shuffle: if true, the data generator shuffels the before each epoch
    '''
   
    def __init__(self, path_ToTrainData, idList, batch_size=1, imagedim=(320,320), n_InputChannels=1,
                 n_classes=3, shuffle=True, augmentation=False):
        self.augmentation= augmentation
        self.idList = idList        
        self.path_ToTrainData = path_ToTrainData
        self.dim = imagedim
        self.batch_size = batch_size
        self.n_InputChannels = n_InputChannels        
        self.n_classes = n_classes
        self.shuffle = shuffle        
        self.indexes = self.getIndexes(idList)
        self.on_epoch_end() 
        self.Image_processor = ImageProcessing()        
        
        
    def getIndexes(self, folderNames):
        'converts lits of ids in from of strings to ints'
        IDs = np.zeros(len(folderNames), dtype=int)
        i = 0        
        for  name in folderNames:
            IDs[i] = int(re.findall('\d+',name)[0])
            i = i+1
        return np.sort(IDs)    
                            
    def on_epoch_end(self):
        'Updates indexes after each epoch'        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __data_generation(self, list_IDs_temp):        
        'Generates data containing batch_size samples loading Data from Train folder' # X : (n_samples, *dim, n_Inputchannels) 
                                                                                        # Y : (batch_size, *dim, n_classes)                                                   
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_InputChannels))
        Y = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_classes))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load x and y data             
            X_loaded = np.load(os.path.join(self.path_ToTrainData , 'Train_RawData', str(ID) +'_raw.npy'))
            Y_loaded = np.load(os.path.join(self.path_ToTrainData , 'Train_LabelData', str(ID) +'_label.npy'))        
            
            # resize (downsample if 512 to original size 256)            
            if X_loaded.shape[0] == 512 or X_loaded.shape[0] == 512:
                X_loaded = self.Image_processor.resize_image(X_loaded, (256, 265))
                Y_loaded = self.Image_processor.resize_image(Y_loaded, (256, 265))
                
            # normlaize input data to range 0-1
            X_loaded = self.Image_processor.normalize(X_loaded)          
            
            # padd images if the dimention is smaller the the defined models 
            # input size 
            if self.dim[0] > X_loaded.shape[0] or self.dim[1] > X_loaded.shape[1]:                
                X_loaded = self.Image_processor.paddimage_new(X_loaded, self.dim)
                Y_loaded = self.Image_processor.paddlabelTensor(Y_loaded,self.dim)
            # copy input grayscale image 3 time to comply with vgg16 input 
            # requirement
            if self.n_InputChannels == 3:   
                if len(X_loaded.shape) < 3:
                    X_loaded = np.array([X_loaded,  X_loaded,  X_loaded])
                    X_loaded = np.moveaxis(X_loaded, 0, -1) 
            else:
                 # add dimension to make it fit in keras function 
                 X_loaded = X_loaded[..., np.newaxis]           

            X[i] = X_loaded
            Y[i] = Y_loaded
        return X, Y
    
    
    def addTransformation(self, X, Y):
        datagen = ImageDataGenerator(shear_range=0.2, rotation_range=2, zoom_range=[0.7,1])    
        datagen_label = ImageDataGenerator(shear_range=0.2, rotation_range=2, zoom_range=[0.7,1])    
        seed = random.randint(1,300)    
        # prepare iterator
        it_raw = datagen.flow(X, batch_size=self.batch_size, seed=seed)
        it_label = datagen_label.flow(Y, batch_size=self.batch_size, seed=seed)
        return it_raw.next(), it_label.next()
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        tmp = self.indexes
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, Y = self.__data_generation(indexes)
        # perfrom augmentation # augmetataion parameters are defined in 
        # addTransformation function
        if self.augmentation:
          X, Y = self.addTransformation(X, Y)
        
        return X, Y     
    
    
class DataGenerator_AbdominalTissue_WithWeightmaps(keras.utils.Sequence):
    '''
    this generator is an extension of the 'DataGenerator_AbdominalTissue' above and load numpy image files to provide them to the U-Net.
    
    Args:
        path_ToTrainData:   path to the image data 
        batch_size:         size of batches, the network will be fed with 
        Imagedim:           the image dimentions (has to be set same as netwrok input dimention)
        n_InputChannels:    the nnumber of channels the input image has (grayscale = 1, color = 3)
        n_classes:          nunmber ob classes the image will be segmented in 
        shuffle:            if true, the data generator shuffels the data every time again before feeding it 
                            to the network so that the images are provided in random sequence 
        modelType:          here not relevant
    '''
    def __init__(self, path_ToTrainData, path_ToWeightMaps, batch_size=1, Imagedim=(320,320), n_InputChannels=1,
                 n_classes=4, shuffle=True, modelType = ""):
        
        'Initialization'
        self.path_ToTrainData = path_ToTrainData
        self.dim = Imagedim
        self.batch_size = batch_size      
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = self.getIndexes(path_ToWeightMaps)
        self.path_ToWeightMaps = path_ToWeightMaps
        self.on_epoch_end() 
                
        if modelType == "VGG16":
            self.n_InputChannels = 3
            self.VGG16 = True
        else:
            self.n_InputChannels = n_InputChannels
            self.VGG16 = False
        
    def getIndexes(self, pathToWeightMaps):
        '''
        This method loops through all the file names at 'pathToWeightMaps' and extracts their IDs 
        Args:
            pathToWeightMaps: path where weightmaps are stored
        '''
        folderNames = os.listdir(pathToWeightMaps)  
        IDs = np.zeros(len(folderNames), dtype=int)
        i = 0        
        for  name in folderNames:
            IDs[i] = int(re.findall('\d+',name)[0])
            i = i+1
        return np.sort(IDs)    
                            
    def on_epoch_end(self):
        'Updates indexes after each epoch'        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))


    def __data_generation(self, list_IDs_temp):        
        'Generates data containing batch_size samples loading Data from Train folder' # X : (n_samples, *dim, n_Inputchannels) 
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_InputChannels))
        out = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_classes*2))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load raw image/images
            X_loaded = np.load(os.path.join(self.path_ToTrainData, 'Train_RawData', str(ID) +'_raw.npy'))
                       
            if self.VGG16 == True:
                X_loaded = np.array([X_loaded,  X_loaded,  X_loaded])
                X_loaded = np.moveaxis(X_loaded, 0, -1) 
            else:
                 # add dimension to mke it fit in keras fit function 
                 X_loaded = X_loaded[..., np.newaxis]
            X[i] = X_loaded
            w = np.load(os.path.join(self.path_ToWeightMaps, str(ID) +'_wmap.npy'))
            y= np.load(os.path.join(self.path_ToTrainData , 'Train_LabelData', str(ID) +'_label.npy'))     
            out[i] = np.concatenate((y, w), axis=-1)            
        return X, out                 

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, out = self.__data_generation(indexes)        
        return X, out


# =========================== TEST DATA Generators ============================

# -------------------------help plot functions -------------------------------
def disp_raw_label(raw_img, overlay_img, name):
    overlayMask(raw_img, overlay_img[:, :, 0], name +'background')
    overlayMask(raw_img, overlay_img[:, :, 1], name +'ssat')
    overlayMask(raw_img, overlay_img[:, :, 2], name +'dsat')
    overlayMask(raw_img, overlay_img[:, :, 3], name +'vat')


def overlayMask(raw, mask, figurename):
    '''
    args:
        raw: rawimage 
        mask: image mask 
        figurename: name in which the resulting image will be displayd
    
    this method overlays the raw image with maks and displays it 
    '''
    plt.figure(figurename)   
    plt.imshow(raw, 'gray')
    plt.imshow(mask, cmap='gray', alpha=0.5) # interpolation='none   

import matplotlib.pyplot as plt
def testDataGenerator():
    print('testing data generator output')
    
    pathToTrainingData = '/media/kwaygo/ymk_HDD/Data/AAT_seg/SampleData/Neonates_extracted/'
    IdlistPath = os.path.join(pathToTrainingData, 'training_IDs.csv')
    ID_list = readCSVToList(IdlistPath)
    # define DataGenerator parameters                        
    params_trainGen = {'path_ToTrainData' : pathToTrainingData,
                       'idList': ID_list,
                       'batch_size': 1,
                       'imagedim': (320,320),          
                       'n_InputChannels': 1,
                       'n_classes': 4,
                       'shuffle': True,
                       'augmentation': False}
    generator = DataGenerator_AbdominalFatSeg(**params_trainGen)
    out = generator.__getitem__(0)
    
    print('X shape is: ' + str(out[0].shape))
    X  = np.squeeze(out[0])
    Y = np.squeeze(out[1])
    plt.imshow(X, 'gray')
    disp_raw_label(X, Y, "dataGen output")
    
def main():
    testDataGenerator()
    
    
if __name__ == '__main__':
    main()    





















# class DataGenerator_AbdominalFatSeg(keras.utils.Sequence):
#     '''
#     this generator load numpy image files to provide them to the Neural Network.
#     resitzing infant data by zerro padding 
    
#     Args:
#         path_ToTrainData : path to the image data                 
#         idList           : list containing the ids for training/validation        
#         batch_size       : Batch size 
#         imagedim         : image dimention shape = (w, h)
#         n_InputChannels  : number of input channels 
#         n_classes        : number of classes
#         shuffle: if true, the data generator shuffels the before each epoch
#     '''
   
#     def __init__(self, path_ToTrainData, idList, batch_size=1, imagedim=(320,320), n_InputChannels=1,
#                  n_classes=3, shuffle=True, augmentation=False):

#         self.augmentation= augmentation
#         self.idList = idList        
#         self.path_ToTrainData = path_ToTrainData
#         self.dim = imagedim
#         self.batch_size = batch_size
#         self.n_InputChannels = n_InputChannels        
#         self.n_classes = n_classes
#         self.shuffle = shuffle        
#         self.indexes = self.getIndexes(idList)
#         self.on_epoch_end() 
#         self.Image_processor = ImageProcessing()        
        
        
#     def getIndexes(self, folderNames):
#         'converts lits of ids in from of strings to ints'
#         IDs = np.zeros(len(folderNames), dtype=int)
#         i = 0        
#         for  name in folderNames:
#             IDs[i] = int(re.findall('\d+',name)[0])
#             i = i+1
#         return np.sort(IDs)    
                            
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'        
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.indexes) / self.batch_size))

#     def __data_generation(self, list_IDs_temp):        
#         'Generates data containing batch_size samples loading Data from Train folder' # X : (n_samples, *dim, n_Inputchannels) 
#                                                                                         # Y : (batch_size, *dim, n_classes)                                                   
#         # Initialization
#         X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_InputChannels))
#         Y = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_classes))
#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # load x and y data             
#             X_loaded = np.load(os.path.join(self.path_ToTrainData , 'Train_RawData', str(ID) +'_raw.npy'))
#             Y_loaded = np.load(os.path.join(self.path_ToTrainData , 'Train_LabelData', str(ID) +'_label.npy'))        
#             # normlaize input data to range 0-1
#             X_loaded = self.Image_processor.normalize_zero_one(X_loaded)          
#             # padd images if the dimention is smaller the the defined models 
#             # input size 
#             if self.dim[0] > X_loaded.shape[0] or self.dim[1] > X_loaded.shape[1]:                
#                 X_loaded = self.Image_processor.paddimage_new(X_loaded, self.dim)
#                 Y_loaded = self.Image_processor.paddlabelTensor(Y_loaded,self.dim)
#             # copy input grayscale image 3 time to comply with vgg16 input 
#             # requirement
#             if self.n_InputChannels == 3:
#                 if len(X_loaded.shape) < 3:
#                     X_loaded = np.array([X_loaded,  X_loaded,  X_loaded])
#                     X_loaded = np.moveaxis(X_loaded, 0, -1) 
#             else:
#                  # add dimension to make it fit in keras function 
#                  X_loaded = X_loaded[..., np.newaxis]           
#             X[i] = X_loaded
#             Y[i] = Y_loaded
#         return X, Y
    
#     def addTransformation(self, X, Y):
#         datagen = ImageDataGenerator(shear_range=0.2, rotation_range=2, zoom_range=[0.6,1])    
#         datagen_label = ImageDataGenerator(shear_range=0.2, rotation_range=2, zoom_range=[0.6,1])    
#         seed = random.randint(1,300)    
#         # prepare iterator
#         it_raw = datagen.flow(X, batch_size=self.batch_size, seed=seed)
#         it_label = datagen_label.flow(Y, batch_size=self.batch_size, seed=seed)
#         return it_raw.next(), it_label.next()
    
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         # Generate data
#         X, Y = self.__data_generation(indexes)
#         # perfrom augmentation # augmetataion parameters are defined in 
#         # addTransformation function
#         if self.augmentation:
#           X, Y = self.addTransformation(X, Y)
        
#         return X, Y     













