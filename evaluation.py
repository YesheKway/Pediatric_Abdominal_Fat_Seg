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
       # define labelnames and set number of classes
       if self.SEGTAST == "VAT_SSAT_DSAT":
           self.labelnames = ("BG", "SSAT", "DSAT", "VAT")
       elif  self.SEGTAST == "SSAT_DSAT":
            self.labelnames = ("BG", "SSAT", "DSAT")
       elif self.SEGTAST == "SAT_VAT":             
            self.labelnames =("BG", "SAT", "VAT")
       self.nClasses = len(self.labelnames)
       if modelPath != "":
           self.load_Model(modelPath)
       self.imageProcessor = ImageProcessing()

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
    
   def calc_FalsePositive(self, groundTruth, prediction):
       'False Positive (FP): we predict a label of 1 (positive), but the true label is 0.'
       FP = np.sum(np.logical_and(prediction == 1, groundTruth == 0))
       return FP
    
   def calc_TruePositiv(self, groundTruth, prediction):
       'True Positive (TP): we predict a label of 1 (positive), and the true label is 1'
       TP = np.sum(np.logical_and(prediction == 1, groundTruth == 1))       
       return TP
   
   def calc_TrueNegative(self, groundTruth, prediction):
       ' True Negative (TN): we predict a label of 0 (negative), and the true label is 0.'
       TN = np.sum(np.logical_and(prediction == 0, groundTruth== 0))
       return TN
    
   def calc_dice(self, true_mask, pred_mask, non_seg_score=1.0):
       """
       Computes the Dice coefficient.
       Args:
           true_mask : Array of arbitrary shape.
           pred_mask : Array with the same shape than true_mask.  
       Returns:
           A scalar representing the Dice coefficient between the two 
           segmentations. 
       """
       assert true_mask.shape == pred_mask.shape       
       true_mask = np.asarray(true_mask).astype(np.bool)
       pred_mask = np.asarray(pred_mask).astype(np.bool)       
       # If both segmentations are all zero, the dice will be 1.
       im_sum = true_mask.sum() + pred_mask.sum()
       if im_sum == 0:
           return non_seg_score
       # Compute Dice coefficient
       intersection = np.logical_and(true_mask, pred_mask)
       return np.round(2. * intersection.sum() / im_sum, 2) 

        
   def calculate_diceForLabels(self, groundtruth, labels):
        """
         this functionn claculates the dice for all labels 
        """
        if self.SEGTAST == "VAT_SSAT_DSAT":
            # dice for background 
            dice_bg = self.calc_dice(groundtruth[:, :, 0], labels[:, :, 0])
            # dice for ssat
            dice_ssat = self.calc_dice(groundtruth[:, :, 1], labels[:, :, 1])
            # dice for dsat
            dice_dsat = self.calc_dice(groundtruth[:, :, 2], labels[:, :, 2])
            # dice for vat
            dice_vat = self.calc_dice(groundtruth[:, :, 3], labels[:, :, 3]) 
            return dice_bg, dice_ssat,  dice_dsat, dice_vat  
        elif self.SEGTAST == "SSAT_DSAT" | "VAT_SAT":            
            # dice for background 
            dice_bg = self.calc_dice(groundtruth[:, :, 0], labels[:, :, 0])
            # dice for ssat
            dice_ssat = self.calc_dice(groundtruth[:, :, 1], labels[:, :, 1])
            # dice for dsat        
            dice_dsat = self.calc_dice(groundtruth[:, :, 2], labels[:, :, 2])    
            return dice_bg, dice_ssat,  dice_dsat   

     
   def calc_BatchDice(self, gtruthBatch, predBatch):  
       """this method calculates the total dice for a batch 
       Return: 
           in our case (bg, ssat, dsat, vat)
       """
       # transpose arrays
       gtruth = np.moveaxis(gtruthBatch, 3, 0)
       pred = np.moveaxis(predBatch, 3, 0)   
       dices = np.zeros((self.nClasses))   
       for n in range(self.nClasses):
           dices[n] = self.calc_dice(gtruth[n], pred[n])
       return dices  
   
    
   def calc_FNFP_batch(self, gtruthBatch, predBatch):  
       'this method calculates the total dice for a batch'
       # transpose arrays
       gtruth = np.moveaxis(gtruthBatch, 3, 0)
       pred = np.moveaxis(predBatch, 3, 0)
       fpfn_rates = np.zeros((self.nClasses,self.nClasses))
       gt_bg_sum = gtruth[0].sum() 
       gt_ssat_sum = gtruth[1].sum()
       gt_dsat_sum = gtruth[2].sum()
       
       if self.SEGTAST == "VAT_SSAT_DSAT":
           gt_vat_sum = gtruth[3].sum()
   
       if self.SEGTAST == "VAT_SSAT_DSAT":
           fpfn_rates[0,0] = self.calc_FalsePositive(gtruth[0],pred[0]) / gt_bg_sum
           fpfn_rates[0,1] = self.calc_FalsePositive(gtruth[1],pred[1]) / gt_ssat_sum
           fpfn_rates[0,2] = self.calc_FalsePositive(gtruth[2],pred[2]) / gt_dsat_sum
           fpfn_rates[0,3] = self.calc_FalsePositive(gtruth[3],pred[3]) / gt_vat_sum   
           fpfn_rates[1,0] = self.calc_FalseNegative(gtruth[0],pred[0]) / gt_bg_sum
           fpfn_rates[1,1] = self.calc_FalseNegative(gtruth[1],pred[1]) / gt_ssat_sum
           fpfn_rates[1,2] = self.calc_FalseNegative(gtruth[2],pred[2]) / gt_dsat_sum
           fpfn_rates[1,3] = self.calc_FalseNegative(gtruth[3],pred[3]) / gt_vat_sum 
       elif  self.SEGTAST == "SSAT_DSAT":
           fpfn_rates[0,0] = self.calc_FalsePositive(gtruth[0],pred[0]) / gt_bg_sum
           fpfn_rates[0,1] = self.calc_FalsePositive(gtruth[1],pred[1]) / gt_ssat_sum
           fpfn_rates[0,2] = self.calc_FalsePositive(gtruth[2],pred[2]) / gt_dsat_sum
           fpfn_rates[1,0] = self.calc_FalseNegative(gtruth[0],pred[0]) / gt_bg_sum
           fpfn_rates[1,1] = self.calc_FalseNegative(gtruth[1],pred[1]) / gt_ssat_sum
           fpfn_rates[1,2] = self.calc_FalseNegative(gtruth[2],pred[2]) / gt_dsat_sum
       return fpfn_rates


   def printAndSaveFinalResults(self, dice_list, fp_list, fn_list, saveSegResults='', childrenData=False):
       """
       Prints final dice results and FP anf FN
       Args: 
           :childrenData: boolen to indicate what data is processed so 
           name can be given to Excel sheet which contains results
       """
       dice_res = np.asarray(dice_list)
       fp_res = np.asarray(fp_list)
       fn_res = np.asarray(fn_list)
       
       # ------ calc total dice, false positive and false negative results  
       # TODO --> reduce code (use dataframes as created in 'if saveSegResults' 
       # below instead of individual variables) 
       
       bg_mean_dice = np.mean(dice_res[:,0])
       bg_std_dice = np.std(dice_res[:,0])   
       bg_mean_fp = np.mean(fp_res[:,0])
       bg_std_fp = np.std(fp_res[:,0])
       bg_mean_fn = np.mean(fn_res[:,0])
       bg_std_fn = np.std(fn_res[:,0])

       ssat_mean_dice = np.mean(dice_res[:,1]) 
       ssat_std_dice = np.std(dice_res[:,1])              
       ssat_mean_fp = np.mean(fp_res[:,1]) 
       ssat_std_fp = np.std(fp_res[:,1])       
       ssat_mean_fn = np.mean(fn_res[:,1]) 
       ssat_std_fn = np.std(fn_res[:,1])     

       dsat_mean_dice = np.mean(dice_res[:,2]) 
       dsat_std_dice = np.std(dice_res[:,2]) 
       dsat_mean_fp = np.mean(fp_res[:,2]) 
       dsat_std_fp = np.std(fp_res[:,2])   
       dsat_mean_fn = np.mean(fn_res[:,2]) 
       dsat_std_fn = np.std(fn_res[:,2])                    

       vat_mean_dice = np.mean(dice_res[:,3]) 
       vat_std_dice = np.std(dice_res[:,3])
       vat_mean_fp = np.mean(fp_res[:,3]) 
       vat_std_fp = np.std(fp_res[:,3])
       vat_mean_fn = np.mean(fn_res[:,3]) 
       vat_std_fn = np.std(fn_res[:,3])
                                
       # ------ print Dice results       
       print('_'*30)
       print('bg:' + str(bg_mean_dice) + '  std:' + str(bg_std_dice))
       print('ssat:' + str(ssat_mean_dice) + '  std:' + str(ssat_std_dice))
       print('dsat:' + str(dsat_mean_dice) + '  std:' + str(dsat_std_dice))
       print('vat:' + str(vat_mean_dice) + '  std:' + str(vat_std_dice))
       # ------ print FP
       print('_'*30)
       print('bg_fp:' + str(bg_mean_fp) + '  std:' + str(bg_std_fp))
       print('ssat_fp:' + str(ssat_mean_fp) + '  std:' + str(ssat_std_fp))
       print('dsat_fp:' + str(dsat_mean_fp) + '  std:' + str(dsat_std_fp))
       print('vat_fp:' + str(vat_mean_fp) + '  std:' + str(vat_std_fp))
       # ------ print FN
       print('_'*30)
       print('bg_fn:' + str(bg_mean_fn) + '  std:' + str(bg_std_fn))
       print('ssat_fn:' + str(ssat_mean_fn) + '  std:' + str(ssat_std_fn))
       print('dsat_fn:' + str(dsat_mean_fn) + '  std:' + str(dsat_std_fn))
       print('vat_fn:' + str(vat_mean_fn) + '  std:' + str(vat_std_fn))
       
       if saveSegResults!= '':  
           df_bg = self.Compute_meanAndstd(dice_res[:,0], fp_res[:,0], 
                                           fn_res[:,0], 'BG')        
           df_ssat = self.Compute_meanAndstd(dice_res[:,1], fp_res[:,1], 
                                             fn_res[:,1], 'SSAT')
           df_dsat = self.Compute_meanAndstd(dice_res[:,2], fp_res[:,2], 
                                             fn_res[:,2], 'DSAT')
           df_vat = self.Compute_meanAndstd(dice_res[:,3], fp_res[:,3], 
                                            fn_res[:,3], 'VAT')
           all_ = pd.concat([df_bg, df_ssat, df_dsat, df_vat], axis=1, sort=False) 
           if childrenData == True:
               Excel_name = 'SegResults_Children.xlsx'
           else:
               Excel_name = 'SegResults_Neonates.xlsx'                                
           all_.to_excel(os.path.join(saveSegResults, Excel_name))  
       
        
   def Compute_meanAndstd(self, dice, fp, fn, name='idk'):
       '''
       computed mean and std for dice, fp and fn rates and returns in 
       pandas dataframe    
       Args:
           :name: is the class name to indicate the class in the dataframe
       '''
       return self.ResultsToDf(np.mean(dice), np.std(dice), np.mean(fp), np.std(fp), np.mean(fn), np.std(fn), name=name)
       
       
   def ResultsToDf(self, dsc_mean, dsc_std, fp_mean, fp_std, fn_mean, fn_std, name='name'):
       '''
       created Dataframe for results
       '''
       data = {name: [dsc_mean, dsc_std, fp_mean, fp_std, fn_mean, fn_std]}
       rownames =['dsc_mean', 'dsc_std', 'fp_mean', 'fp_std', 'fn_mean', 'fn_std']
       df = pd.DataFrame(data, rownames)
       df = df.round(3)     
       return df

       
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
            self.evaluateModel(path_toModelFile, csv_path_neonates, config['infant_data_path'], loss=config['loss_function'], saveOverallResults=abspath)
            # evaluate model on childrne data 
            self.evaluateModel(path_toModelFile, csv_path_children, config['children_data_path'], loss=config['loss_function'], saveOverallResults=abspath, childrenData = True)    

        



   def evaluateModel(self, modelPath, csv_path_evaluationIDs, pathToData, loss='dice_coef_multilabel', saveOverallResults='', childrenData = False):
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
        print('-'*30)
        print('Evaluating Dataset...')
        print('-'*30) 
        
        # store evaluation results                 
        dice_list = list()       
        fp_list = list()
        fn_list = list()
                                     
        for subject in all_subjects:          
            # get raw and label images for ROI         
            raw_data_path = os.path.join(pathToRawData, subject + "_raw.npy")           
            label_data_path = os.path.join(pathToLabelData, subject + "_label.npy")           
            raw_ROI = np.load(raw_data_path)
            labels_ROI_groundTruth = np.load(label_data_path)             
            

            if self.input_shape_model[0] > raw_ROI.shape[1] or self.input_shape_model[1] > raw_ROI.shape[2]:                
                raw_ROI = self.preprocessing.paddBatch(raw_ROI, (self.input_shape_model[0], 
                                                                 self.input_shape_model[1]))  
                labels_ROI_groundTruth = self.preprocessing.paddLabelBatch(labels_ROI_groundTruth,
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
                                       
            # calculate false possitive and false negative 
            fpfn_rates_percentage = self.calc_FNFP_batch(labels_ROI_groundTruth, pred) 
            fp_list.append([fpfn_rates_percentage[0, 0], fpfn_rates_percentage[0, 1], fpfn_rates_percentage[0, 2], fpfn_rates_percentage[0, 3]])
            fn_list.append([fpfn_rates_percentage[1, 0], fpfn_rates_percentage[1, 1], fpfn_rates_percentage[1, 2], fpfn_rates_percentage[1, 3]])
            
            # calculate dice
            dices = self.calc_BatchDice(labels_ROI_groundTruth, pred)
            dice_list.append(dices)                                                                             
    #            print(subject,'dice is:')
    #            self.printDiceResults(dices)           
        self.printAndSaveFinalResults(dice_list, fp_list, fn_list, saveSegResults=saveOverallResults, childrenData=childrenData)   


    
# def EvaluateTrainedModel():
#     pathToSubjects = '/home/mrsmig/sda6/Yeshe/Data/Infant/TotalTrainingSubj'    
#     pathToModel = "/home/mrsmig/sda6/Yeshe/TrainedModels/InfantModels/BachelorModels/model2-improvement-02-2479.62.h5"

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
    
    
# import yaml
# import argparse    

# def main():
    
    
#     print('starting program ...')
#     parser = argparse.ArgumentParser(prog='MedSeg')
#     parser.add_argument('-config', '--config_path', type=str, default='./config/default_config.yaml', help='Configuration file defining training and evaluation parameters')
#     args = parser.parse_args()
#     with open(args.config_path) as file:
#         config = yaml.load(file, Loader=yaml.FullLoader)
    
# #     ev = Evaluater()        
# #     ev.EvaluateModels(config)
    
    
    
    
# if __name__ == "__main__":
#     main()