#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:07:53 2020

@author: ymk
"""
import SimpleITK as sitk
import numpy as np
from cv2 import resize


class ImageProcessing():
    
    
    def normalize_zero_one(self, image):        
        """
        normalizes image pixel values to the range from 0 to 1
        """
        return (image- np.min(image))/(np.max(image)-np.min(image)) 
    
    def normalize(self, image):        
         """
         normalizes image pixel values to the range from 0 to 1
         """
         return image/image.max()  

    def aRGMAX(self, probabilitymaps):         
        pred_amax = np.argmax(probabilitymaps, axis=3)       
        bg =  (pred_amax == 0).astype(int) 
        ssat = (pred_amax == 1).astype(int) 
        dsat = (pred_amax == 2).astype(int) 
        vat = (pred_amax == 3).astype(int) 
        return np.moveaxis(np.stack((bg, ssat, dsat, vat)), 0, -1)           

    
    def biasFielCorrectionBatch(self, img_batch):
         """
         Args:
             img_batch: shape = (batchsite, row, col)
         """
         corrected = np.zeros_like(img_batch)
         for i in range(len(img_batch)):
             corrected[i] = self.biasFieldCorrection(img_batch[i])
         return np.array(corrected)    
     
    def biasFieldCorrection(self, img):
         """
         this function applies N4 bias field correction MR image 
         """
         img = img.astype(np.int16)
         img = sitk.GetImageFromArray(img)
         maskImage = sitk.OtsuThreshold(img, 0,1,20)  
         img = sitk.Cast(img,sitk.sitkFloat32)
         corrector = sitk.N4BiasFieldCorrectionImageFilter();
         img_corrected = corrector.Execute(img,maskImage)                
         return sitk.GetArrayFromImage(img_corrected)
    
    
    def pad_with(self, vector, pad_width, iaxis, kwargs):
        '''
        Help function to define padd calue 
        '''
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
    
    
    def paddimage_new(self, img, new_shape, paddValue=0):
        '''
        This function padds the batch by filling blanks to top and bottom         
        
        Parameters
        ----------
        img : numpy
            DESCRIPTION.
        new_shape :
            DESCRIPTION. (new_hight, new_width).
        col : TYPE, optional
            DESCRIPTION. The default is 'constant'.
        Returns
        -------
        None.
    
        '''    
        input_shape = img.shape
        delta_w = new_shape[1] - input_shape[1]
        delta_h = new_shape[0] - input_shape[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        padded_tensor = np.pad(img, ((top,bottom),(left,right)), self.pad_with, padder=paddValue)
        return padded_tensor  
    
    def paddlabelTensor(self, lableTensor, new_shape):
          '''
          Args:
              lableTensor: must be in shape (w, h, channels), where channel 0 
              must be background label
          '''
          w, h, c = lableTensor.shape
          new_tensor = np.zeros((new_shape[0], new_shape[1], c))
          for i in range(c):
              img = lableTensor[:,:,i]
              if i == 0:
                  new_tensor[:,:,i] = self.paddimage_new(img, new_shape, 1)        
              else:
                  new_tensor[:,:,i] = self.paddimage_new(img, new_shape, 0)        
          return new_tensor
      
    def resize_image(self, img, newsize):
        return resize(img, newsize)
    
    def resize_labels(self, label_tensor, newsize):
        new = np.empty(newsize, newsize, label_tensor.shape[-1])
        for i in range(label_tensor.shape[-1]):
            new[:, :, i] = self.resize_image(label_tensor[:, :, 1], newsize) 
        return new
        
















          