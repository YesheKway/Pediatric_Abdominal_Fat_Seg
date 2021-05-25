#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 07:24:45 2020
@author: Yeshe Kway
"""
from Utils import class_weighted_pixelwise_crossentropy, tversky_loss
from Utils import init_GPU_Session, readCSVToList
from ImageProcessing import ImageProcessing
from Utils import sort_dir_aphanumerical    
from Utils import dice_coef_multilabel
from keras.models import load_model
from tqdm import tqdm
import pandas as pd
import numpy as np
import os 

class Evaluater():

   def __init__(self, SEGTAST='VAT_SSAT_DSAT', modelPath=''):
       
       self.SEGTAST = SEGTAST
       self.metrice_names = ["dice", "fp", "fn", "tp", "tn"]
       self.imageProcessor = ImageProcessing()
       self.models_mean_performance_neonates = pd.DataFrame()
       self.models_mean_performance_children = pd.DataFrame()
       # define labelnames and set number of classes
       if self.SEGTAST == "VAT_SSAT_DSAT":
           self.labelnames = ("bg", "ssat", "dsat", "vat")
       self.nClasses = len(self.labelnames)
       
       self.metric_func_dic = {'fp': self.calc_FalsePositive, 
                               'fn': self.calc_FalseNegative, 
                               'tp': self.calc_TruePositiv, 
                               'tn': self.calc_TrueNegative, 
                               'dice': self.calc_dice}
       if modelPath != "":
           self.load_Model(modelPath)


# ================== define evaluation metrices ================================'

   def calc_FNFPTPTN(self, groundTruth, prediction):
       '''
       :param self:
       :param groundTruth: groundtruth data in 
                           shape = (batch, img_width, img_height, nub_classes)
       :param prediction: prediction array we want to evaluate agains 
                          groundtruth  
                          shape = (batch, img_width, img_height, nub_classes)
       '''
       FN = self.calc_FalseNegative(groundTruth, prediction)
       FP = self.calc_FalsePositive(groundTruth, prediction)
       TP = self.calc_TruePositiv(groundTruth, prediction)
       TN = self.calc_TrueNegative(groundTruth, prediction)
       print ('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))

   def calc_FalseNegative(self, groundTruth, prediction):
       'False Negative (FN): we predict a label of 0 (negative), but the true label is 1.'
       FN = np.sum(np.logical_and(prediction == 0, groundTruth == 1))
       return FN
   
   def calc_FalsePositive_rate(self, FP, TN):
       return FP/(FP+TN)    
   
   def calc_FalsePositive(self, groundTruth, prediction):
       'False Positive (FP): we predict a label of 1 (positive), but the true label is 0.'
       FP = np.sum(np.logical_and(prediction == 1, groundTruth == 0))
       return FP
   
   def calc_FalseNegative_rate(self, FN, TP):
       i = FN/(FN+TP)
       return FN/(FN+TP)
   
   def calc_TruePositiv(self, groundTruth, prediction):
       'True Positive (TP): we predict a label of 1 (positive), and the true label is 1'
       TP = np.sum(np.logical_and(prediction == 1, groundTruth == 1))       
       return TP
   
   def calc_TrueNegative(self, groundTruth, prediction):
       ' True Negative (TN): we predict a label of 0 (negative), and the true label is 0.'
       TN = np.sum(np.logical_and(prediction == 0, groundTruth== 0))
       return TN
   
   def calc_pix_accuracy(self, TP, TN, FP, FN):
       return (TP + TN) / (TP + TN + FP + FN)
   
   def calc_precision(self, TP, FP):
       # compute Precision (Positive predictive value)
       return TP/(TP+FP)
   
   def calc_metrices(self, groundTruth, prediction):  
        '''
        this function computes FP, FP, TN, TP, positive predictive, FP rate, FN rate, 
        dice similarity coefficient and accuracy
        '''
        gtruth = np.moveaxis(groundTruth, -1, 0)
        pred = np.moveaxis(prediction, -1, 0)
        # compute all metrices for all labels
        computed_metrice_dic = {}
        for i, label_name in enumerate(self.labelnames):
            for metric_key in self.metric_func_dic:
                computed_metrice_dic[label_name+'_'+ metric_key] = self.metric_func_dic[metric_key](gtruth[i], pred[i])        
        
        # compute accuracy, positive predictive value, false positive/negative rate 
        for i, label_name in enumerate(self.labelnames):

            computed_metrice_dic[label_name + '_fn_rate'] = self.calc_FalseNegative_rate(computed_metrice_dic[label_name + '_fn'],
                                                                                  computed_metrice_dic[label_name + '_tp'])
            computed_metrice_dic[label_name + '_fp_rate'] = self.calc_FalsePositive_rate(computed_metrice_dic[label_name + '_fp'],
                                                                                  computed_metrice_dic[label_name + '_tn'])
            computed_metrice_dic[label_name + '_precision'] = self.calc_precision(computed_metrice_dic[label_name + '_tp'],
                                                                                  computed_metrice_dic[label_name + '_fp'])
            computed_metrice_dic[label_name + '_accuracy'] =  self.calc_pix_accuracy(computed_metrice_dic[label_name + '_tp'],
                                                                                     computed_metrice_dic[label_name + '_tn'],
                                                                                     computed_metrice_dic[label_name + '_fp'],
                                                                                     computed_metrice_dic[label_name + '_fn'])
        return computed_metrice_dic   
    
   def calc_dice(self, groundTruth, prediction, non_seg_score=1.0):
       """
       Computes the Dice coefficient.
       Args:
           true_mask : Array of arbitrary shape.
           pred_mask : Array with the same shape than true_mask.  
       Returns:
           A scalar representing the Dice coefficient between the two 
           segmentations. 
       """
       assert groundTruth.shape == prediction.shape       
       groundTruth = np.asarray(groundTruth).astype(np.bool)
       prediction = np.asarray(prediction).astype(np.bool)       
       # If both segmentations are all zero, the dice will be 1.
       im_sum = groundTruth.sum() + prediction.sum()
       if im_sum == 0:
           return non_seg_score
       # Compute Dice coefficient
       intersection = np.logical_and(groundTruth, prediction)
       return 2. * intersection.sum() / im_sum 

        
     
   def calc_BatchDice(self, gtruthBatch, predBatch):  
       """this method calculates the total dice for a batch 
       Return: 
           in our case (bg, ssat, dsat, vat)
       """
       gtruth = np.moveaxis(gtruthBatch, 3, 0)
       pred = np.moveaxis(predBatch, 3, 0)   
       dices = np.zeros((self.nClasses))   
       for n in range(self.nClasses):
           dices[n] = self.calc_dice(gtruth[n], pred[n])
       return dices  

       
   def load_Model(self, pathToModel, loss='dice_coef_multilabel'):
       '''
       load model into Evaluation Manager:
           Args:
               modelFile: absolute path to model
               loss: arguments are tvs, cwpc or dice 
       '''
       if loss == 'dice_coef_multilabel':
           self.model = load_model(pathToModel, custom_objects={'dice_coef_multilabel': dice_coef_multilabel})
       elif loss == 'class_weighted_pixelwise_crossentropy':
           self.model = load_model(pathToModel, 
                                   custom_objects={ 'class_weighted_pixelwise_crossentropy': class_weighted_pixelwise_crossentropy})
       elif loss == 'tversky':       
           self.model = load_model(pathToModel, 
                                   custom_objects={ 'tversky_loss': tversky_loss})
       else: 
           print('please specify loss function when clling loadModel from Evaluation_Manager')
   
       self.input_shape_model = np.delete(self.model.layers[0].input_shape, 0)
           
   def get_BestModel_Name(self, main_dir):
        model_list = list()
        for file in os.listdir(main_dir):
            if file.endswith(".h5"):
                model_list.append(file)
        return sort_dir_aphanumerical(model_list)[-1]
    
   def predict(self, img):
       img = self.imageProcessor.normalize(img)         
       if self.input_shape_model[-1]> 1:
            img = np.array([img,  img, img])
            img = np.moveaxis(img, 0, -1)  
            img = img[np.newaxis,...]
       else:
            img = img[..., np.newaxis]
            img = img[np.newaxis, ...]
       return self.imageProcessor.aRGMAX(self.model.predict(img))       
       
   def EvaluateModels(self, config):
        '''
        This function evaluates multiple models which were previously trained by 
        "k_fold_training" function in trianing.py
        Args:
            :pathToModels: path to dir where models are stored at 
            directory must be as follows:
    
                ./dst/models
                    |
                    |---model1
                    |     |
                    |     |---model.h5
                    |     |---validation_IDs.csv
                    |     |---train_IDs.csv
                    |
                    |---model2
                    |     |
                    |     |---model.h5
                    |     |---validation_IDs.csv
                    |     |---train_IDs.csv
                    .
                    .
                    .
        '''
        pathToModels = os.path.join(config['dst'], 'models')                
        modelFolders = os.listdir(pathToModels)
        # loop through all folders and evaluate model
        for modelFolder in tqdm(modelFolders):        
            # define paths 
            abspath = os.path.join(pathToModels, modelFolder)
            csv_path_children = os.path.join(abspath, 
                                             'children_validationIDs.csv')
            csv_path_neonates = os.path.join(abspath, 
                                             'neonatal_validationIDs.csv')
            path_toModelFile = os.path.join(abspath, self.get_BestModel_Name(abspath))
            # evaluate model on neonatal data         
            self.evaluateModel(path_toModelFile, csv_path_neonates, config['infant_data_path'], loss=config['loss_function'], dst=abspath)
            # evaluate model on childrne data 
            self.evaluateModel(path_toModelFile, csv_path_children, config['children_data_path'], loss=config['loss_function'], dst=abspath, childrenData = True)    
        self.models_mean_performance_neonates.to_excel(os.path.join(pathToModels, 'All_Model_mean_neonates_2.xlsx'))
        self.models_mean_performance_children.to_excel(os.path.join(pathToModels, 'All_Model_mean_children_2.xlsx'))

   def evaluateModel(self, modelPath, csv_path_evaluationIDs, pathToData, loss='dice_coef_multilabel', dst='', childrenData = False):
        """'''
        Args:
            :neonate: True to evaluate on children if False on neonatal
            :saveOverallResults: path to which excel with results will be saved to 
        This function evaluates models on neonatal as well as infant data 
        Args:
           :data_path: path to evaluation data 
           :imgProcessing: if enabled thresholds the prediction output   
           :childrenData: if True images will be reshaped, zero padded to 
           shape 512x512
           :saveOverallResults: if a path is given, a exel with mean and SD info 
           for all classes is save as SegResults.xlsx
        """      
        
        init_GPU_Session()    
        # load model     
        self.load_Model(modelPath, loss)
        
        # define path to raw an gt data 
        pathToRawData = os.path.join(pathToData, 'Train_RawData')
        pathToLabelData = os.path.join(pathToData, 'Train_LabelData')       
        
        # get ids for evaluation 
        all_subjects = readCSVToList(csv_path_evaluationIDs) 
        
        # dataframe to store computed metrice values for all subjects 
        model_performance = pd.DataFrame()                             
        
        for subject in all_subjects:          
            # get raw and label images for ROI         
            raw_data_path = os.path.join(pathToRawData, subject + "_raw.npy")           
            label_data_path = os.path.join(pathToLabelData, subject + "_label.npy")           
            raw_ROI = np.load(raw_data_path)
            labels_ROI_groundTruth = np.load(label_data_path)             
            # reshape if image size is smaller then model input requirement
            if self.input_shape_model[0] > raw_ROI.shape[1] or self.input_shape_model[1] > raw_ROI.shape[2]:                
                raw_ROI = self.imageProcessor.paddBatch(raw_ROI, (self.input_shape_model[0],  self.input_shape_model[1]))  
                labels_ROI_groundTruth = self.imageProcessor.paddLabelBatch(labels_ROI_groundTruth,
                                                                           (self.input_shape_model[0],
                                                                            self.input_shape_model[1]))           
                pred = np.empty((len(raw_ROI), self.input_shape_model[0],
                                 self.input_shape_model[1], self.nClasses))
            else:
                pred = np.empty((len(raw_ROI), len(raw_ROI[1]), len(raw_ROI[2]), 
                                 self.nClasses))
            #make prediction for all images
            for n in range(0,len(raw_ROI)):          
                pred[n] = self.predict(raw_ROI[n])
            # compute evaluation metrices                   
            metrice_row_dic = self.calc_metrices(labels_ROI_groundTruth, pred)
            metrice_row_series = pd.Series(metrice_row_dic)
            metrice_row_series.name = subject
            model_performance = model_performance.append(metrice_row_series) 
        self.SaveResults(model_performance, dst, childrenData=childrenData)    
        # self.printAndSaveFinalResults(dice_list, fp_list, fn_list, saveSegResults=saveOverallResults, childrenData=childrenData)   


   def SaveResults(self, model_performance, dst, childrenData=False):
       # means = model_performance.mean()
       # means.name = 'mean'
        # stds = model_performance.std()
       # stds.name = 'std'    
       # comb = pd.concat([means, stds], axis=1)
       means = model_performance.mean()
       means.name = 'mean'
       stds = model_performance.std()
       stds.name = 'std'    
       median = model_performance.median()
       median.name = 'median'    
       quantile = model_performance.quantile([.1, .25, .5, .75], axis = 0)
       comb = pd.concat([means, stds, median, quantile], axis=1)

       # extra_stats = model_performance.describe()
       
       if childrenData == True:
           Excel_name_ending = '_Children.xlsx'
           self.models_mean_performance_children = self.models_mean_performance_children.append(means, ignore_index=True) 
       else:
           Excel_name_ending = '_Neonates.xlsx'                                
           self.models_mean_performance_neonates = self.models_mean_performance_neonates.append(means, ignore_index=True) 
       
       # extra_stats.to_excel(os.path.join(dst, 'Extra_stats' + Excel_name_ending))      
       model_performance.to_excel(os.path.join(dst, 'All_subj_volume_performance_2' + Excel_name_ending))      
       comb.to_excel(os.path.join(dst, 'Mean_performance_TEST' + Excel_name_ending))      
    



# def EvaluateTrainedModel():
    # pathToSubjects = '/home/mrsmig/sda6/Yeshe/Data/Infant/TotalTrainingSubj'    
    # pathToModel = "/home/mrsmig/sda6/Yeshe/TrainedModels/InfantModels/BachelorModels/model2-improvement-02-2479.62.h5"
    #     ev = Evaluation_Manager()
#     ev.loadModel(pathToModel)
#     ev.setMODEVGG(True)
#     ev.evaluateOnTestData(pathToSubjects, SaveSegmentationAndCreateExcel=False)


# def startMultipleModelEvaluation():
    # ------------ local paths --------------
    # pathToModels = '/home/kwaygo/Documents/Abdominal_Adipose_Segmentation/TrainedModels/tmp'
    # pathToImagingVolumes_Children = '/app/Data/CombinedMultipleTraining/ChildrenVolumes'
    # pathToImagingVolumes_Neonates =  '/app/Data/NeonateData/Volumes'
    # ---------- server paths --------------
    # pathToModels = '/app/TrainedModels/CombinedModel_Multitraining/models'
    # pathToImagingVolumes_Children = '/app/Data/CombinedMultipleTraining/ChildrenVolumes'
    # pathToImagingVolumes_Neonates =  '/app/Data/NeonateData/Volumes'
    # ev = Evaluater()
    # ev.EvaluateModels(config)

def local():
    print('starting program ...')
    config_path = '/media/kwaygo/ymk_HDD1/Documents/Manuscripts/AAT_seg/Paper-Models/Presented10Folds/10_fold_TrainingResults_512_Weights/Second_Evaluation/10_fold_512_Weights/default_config_512.yaml'
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    ev = Evaluater()        
    ev.EvaluateModels(config)    

import yaml
import argparse    

def main():
    local()
    
    # print('starting program ...')
    # parser = argparse.ArgumentParser(prog='AbdoSeg')
    # parser.add_argument('-config', '--config_path', type=str, default='./config/default_config.yaml', help='Configuration file defining training and evaluation parameters')
    # args = parser.parse_args()
    # with open(args.config_path) as file:
    #     config = yaml.load(file, Loader=yaml.FullLoader)
    # ev = Evaluater()        
    # ev.EvaluateModels(config)
    
    
if __name__ == "__main__":
    main()