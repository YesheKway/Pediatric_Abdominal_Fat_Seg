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
                 n_classes=3, shuffle=True, augmentation=False, biasCorrection=True):

        self.augmentation= augmentation
        self.idList = idList        
        self.biasCorrection = biasCorrection
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
            
            # get the data             
            X_loaded = np.load(os.path.join(self.path_ToTrainData , 'Train_RawData', str(ID) +'_raw.npy'))
            Y_loaded = np.load(os.path.join(self.path_ToTrainData , 'Train_LabelData', str(ID) +'_label.npy'))        
                
            if self.biasCorrection:
                X_loaded = self.Image_processor.biasFieldCorrection(X_loaded)
            
            X_loaded = self.Image_processor.normalize(X_loaded)          
            
            if self.dim[0] > X_loaded.shape[0] or self.dim[1] > X_loaded.shape[1]:                
                X_loaded = self.Image_processor.paddimage_new(X_loaded, self.dim)
                Y_loaded = self.Image_processor.paddlabelTensor(Y_loaded,self.dim)

            if self.n_InputChannels == 3:
                if len(X_loaded.shape) < 3:
                    X_loaded = np.array([X_loaded,  X_loaded,  X_loaded])
                    X_loaded = np.moveaxis(X_loaded, 0, -1) 
            else:
                 # add dimension to make it fit in keras function 
                 X_loaded = X_loaded[..., np.newaxis]           
            if self.augmentation:
                if ID <=17992:
                    X_loaded = X_loaded[np.newaxis,...]  
                    Y_loaded  = Y_loaded[np.newaxis,...]   
                    X_loaded, Y_loaded = self.addTransformation(X_loaded, Y_loaded)
                    X_loaded = np.squeeze(X_loaded)
                    Y_loaded = np.squeeze(Y_loaded)
            X[i] = X_loaded
            Y[i] = Y_loaded
        return X, Y
    
    
    def addTransformation(self, X, Y):
        datagen = ImageDataGenerator(shear_range=0.2, rotation_range=2, zoom_range=[0.6,1])    
        datagen_label = ImageDataGenerator(shear_range=0.2, rotation_range=2, zoom_range=[0.6,1])    
        seed = random.randint(1,300)    
        # prepare iterator
        it_raw = datagen.flow(X, batch_size=self.batch_size, seed=seed)
        it_label = datagen_label.flow(Y, batch_size=self.batch_size, seed=seed)
        return it_raw.next(), it_label.next()
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, Y = self.__data_generation(indexes)
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





