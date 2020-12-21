# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 04:15:42 2018
@author: Yeshe Kway
"""
from keras.models import Model
from keras.layers.convolutional import Convolution2DTranspose
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Conv3D
from keras.layers import BatchNormalization, Dropout
from keras.layers import Flatten, Dense, Lambda, multiply, Add, UpSampling2D
from keras.layers.core import  Activation
from keras import backend as K
from keras.layers.merge import concatenate
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from Utils import class_weighted_pixelwise_crossentropy
import tensorflow as tf   
from keras.engine.topology import Layer
import keras

'============================  Customized layers  ============================'

class downSampleLayer(Layer):
    """
    Bilinear Downsamloing layer
    """
    def __init__(self, size=(20,20), interpolation= 'bilinear',  **kwargs):
        self.size = size 
        self.interpolation = interpolation 
        self.output_sha = size 
        super(downSampleLayer, self).__init__(**kwargs)
#        super(downSampleLayer, self).__init__(**kwargs)
    
#    def build( self, input_shape):
#        super(downSampleLayer, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        b, w, h, c = input_shape
        return (b, self.size[0], self.size[1], c)
   
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'size': self.size,
            'interpolation': self.interpolation,
        })
        return config 
    
    def call(self, inputs):
#        resized =  tf.image.resize(inputs, self.size, method=self.interpolation, preserve_aspect_ratio=True)
        resized =  tf.image.resize(inputs, self.size, method= tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True)
#        resized = K.resize_images(inputs, self.size[0], self.size[1], 'channels_last', self.interpolation)
        return resized


class softmaxLayer(Layer):
    '''
    SoftmaxLayer
    '''
    def call(self, inputs):
        clip_actv = tf.clip_by_value(inputs, 10e-8, 1.-10e-8)
        logit_y_pred = K.log(clip_actv)
        return logit_y_pred
                

class dilatedConvBlock(keras.layers.Layer):
    '''
    dilated convolutional block performs 3 convolutional operations and 
    concadenates their outputs 
    '''
    
    def __init__(self, nFilters, dilationRates):
        """
        Args:
            dilationRates: rates for conv1, conv2 and conv3 need to be past like 
            (r1,r2,r3)
        """
        super(dilatedConvBlock, self).__init__()
        self.nFilters= nFilters
        self.dilationRates = dilationRates
    def call(self, X):
        conv1 = keras.layers.Conv2D(filters=self.nFilters, kernel_size=(3,3), 
                                 activation='relu', padding='same', 
                                 dilation_rate = self.dilationRates[0])(X)

        conv2 = keras.layers.Conv2D(filters=self.nFilters, kernel_size=(3,3), 
                                 activation='relu', padding='same', 
                                 dilation_rate = self.dilationRates[0])(X)        
        conv3 = keras.layers.Conv2D(filters=self.nFilters, kernel_size=(3,3), 
                                 activation='relu', padding='same', 
                                 dilation_rate = self.dilationRates[0])(X)        
#        
        return tf.concat([conv1, conv2, conv3], axis=3)


'============================  Unet Manager  ============================'
   
class Unet_Designer():
   
    def __init__(self):        
        self.axis = 3  # for concadination and batchnormalisation 

    def get_model(self, config):
       
        if config['model_type'] == 'Unet_VGG16_upSampling_extramerge':
            return self.get_VGG16_Unet_upSampling_extramerge(config['img_dim'], 
                                                             output_channels= config['n_classes'], 
                                                             drop_out=float(config['dropout']), 
                                                             batch_Norm=config['batch_normalization'], 
                                                             interpolation=config['interpolation'])
        if config['model_type'] == 'Unet_base':
            config['img_dim'].append(config['n_input_channels']), 
            input_shape = (config['img_dim'][0], config['img_dim'][1], 
                           config['img_dim'][2])
            return self.get_Unet(input_shape, output_channels= config['n_classes'],
                                 drop_out=float(config['dropout']),
                                 batch_Norm=config['batch_normalization'])        

# ========================= define customized layers =========================
            
    def bn_conv2D(self, n_filters, kernel_size, input_layer):
        """
        2D Convolution with batchnormalization, Activation is ReLu
        Args:
            n_filters:      Number of Filters 
            kernel_size:    one integer defining kernel size             
            input_layer:    previous keras / tf 
        Return:
            conv:           Output layer 
        """
        conv = Conv2D(n_filters, (kernel_size, kernel_size), padding='same')(input_layer)
        conv = BatchNormalization(axis=self.axis)(conv)
        conv = Activation('relu')(conv)
        return conv

    def double_conv2D(self, n_filters, kernel_size, layer, batch_norm=True, dropout=0.0):
        """
        double 2D Convolution Layer
        Args:
            n_filters:   Number of Filters 
            kernel_size: integer defining kernel size (symetric)
            layer: input layer -> previous layer 
            batch_norm:  Batchnormalization Default is True 
            dropout:     Dropout rate Default is 0.0 
        """
        if batch_norm is True:
            conv1 = self.bn_conv2D(n_filters, kernel_size, layer)
            conv2 = self.bn_conv2D(n_filters, kernel_size, conv1)
        else:
            conv1 = Conv2D(n_filters, (kernel_size, kernel_size), activation='relu', padding='same')(layer)
            conv2 = Conv2D(n_filters, (kernel_size, kernel_size), activation='relu', padding='same')(conv1)
        if dropout > 0.0:
            conv2 = Dropout(dropout)(conv2)            
        return conv2

    def conv2DTransp_bn(self, n_filters, kernel_size, input_layer): 
        """
        2D Transposed Convolutional Layer with BatchNormalization
        Args:
            n_filters:      Number of Filters 
            kernel_size:    one integer defining kernel size             
            input_layer:    previous keras / tf 
        Return:
            upconv:         Output layer 
        """              
        upconv = Convolution2DTranspose(n_filters, (kernel_size, kernel_size), strides=(2, 2), use_bias= False)(input_layer)        
        upconv = BatchNormalization(axis=self.axis)(upconv)
        upconv = Activation('relu')(upconv)   
        return upconv    

    def SubpixelConv2D(self, input_shape, scale=4, name='subpixel'):
        """
        Keras layer to do subpixel convolution.
        NOTE: Tensorflow backend only. Uses tf.depth_to_space
        Ref:
            [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
                Shi et Al.
                https://arxiv.org/abs/1609.05158
        Args:
            :param input_shape tensor shape, (batch, height, width, channel)
            :param scale: upsampling scale. Default=4
        :return:
        """
        # upsample using depth_to_space
        def subpixel_shape(input_shape):
            dims = [input_shape[0],
                    input_shape[1] * scale,
                    input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))]
            output_shape = tuple(dims)
            return output_shape    
        def subpixel(x):
            return tf.depth_to_space(x, scale)        
        return Lambda(subpixel, output_shape=subpixel_shape, name=name)
        
# ==================================== Original Unet structure ================
    def get_Unet(self, img_dim=(320,320, 1), output_channels=4,
                 drop_out=0.0, batch_Norm=True, n_filters = 64):
        
        # define Input layer with input image dim 320x320xchannels
        inputs = Input((img_dim[0], img_dim[1], img_dim[2]))

        # downsamping        
        conv_1 = self.double_conv2D(n_filters, 3, inputs, batch_norm=batch_Norm, dropout=drop_out)
        pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

        conv_2 = self.double_conv2D(n_filters*2, 3, pool_1, batch_norm=batch_Norm, dropout=drop_out)
        pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

        conv_3 = self.double_conv2D(n_filters*4, 3, pool_2, batch_norm=batch_Norm, dropout=drop_out)
        pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

        conv_4 = self.double_conv2D(n_filters*8, 3, pool_3, batch_norm=batch_Norm, dropout=drop_out)
        pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

        conv_5 = self.double_conv2D(n_filters*16, 3, pool_4, batch_norm=batch_Norm, dropout=drop_out)

        # ------- UpSampling ---------
        upconv_1 = self.conv2DTransp_bn(n_filters*8, 2, conv_5)
        conca_1 = concatenate([conv_4,upconv_1], axis = self.axis)
        conv_6 = self.double_conv2D(n_filters*8, 3, conca_1, batch_norm=batch_Norm, dropout=drop_out)

#        upconv_2 = Convolution2DTranspose(n_filters*4, (2, 2), strides=(2, 2), use_bias=USE_BIAS)(conv_6)
        upconv_2 = self.conv2DTransp_bn(n_filters*4, 2, conv_6)
        conca_2 = concatenate([conv_3,upconv_2], axis= self.axis)
        conv_7 = self.double_conv2D(n_filters*4, 3, conca_2, batch_norm=batch_Norm, dropout=drop_out)

#        upconv_3 = Convolution2DTranspose(n_filters*2, (2, 2), strides=(2, 2), use_bias=USE_BIAS)(conv_7)
        upconv_3 = self.conv2DTransp_bn(n_filters*2, 2, conv_7)
        conca_3 = concatenate([conv_2,upconv_3], axis= self.axis)        
        conv_8 = self.double_conv2D(n_filters*2, 3, conca_3, batch_norm=batch_Norm, dropout=drop_out)

        #upconv_4 = Convolution2DTranspose(n_filters, (2, 2), strides=(2, 2), use_bias=USE_BIAS)(conv_8)
        upconv_4 = self.conv2DTransp_bn(n_filters, 2, conv_8)
        conca_4 = concatenate([conv_1,upconv_4], axis= self.axis)
        conv_9 = self.double_conv2D(n_filters, 3, conca_4, batch_norm=batch_Norm, dropout=drop_out)

        output = Conv2D(output_channels, (1, 1))(conv_9)
        output = Activation('softmax', name='main_output')(output)

        model = Model(inputs, output, name="Unet")
        return model       
        
# ============================ FINAL model used in the papaer =================
    def get_VGG16_Unet_upSampling_extramerge(self, img_dim=(320, 320), output_channels=4,
                                             drop_out=0.0, batch_Norm=True, USE_BIAS = False, interpolation='bilinear'):
        
        n_filters = 64
        inputs = Input((img_dim[0], img_dim[1], 3))
        #get VGG16
        vgg16 = VGG16(input_tensor=inputs, include_top=False)
        for l in vgg16.layers:
            l.trainable = True
        out_vgg16 = vgg16(inputs)
    
        #get vgg layer outputs    
        block1_conv2 = vgg16.get_layer("block1_conv2").output    
        block2_conv2 = vgg16.get_layer("block2_conv2").output
        block3_conv3 = vgg16.get_layer("block3_conv3").output      
        block4_conv3 = vgg16.get_layer("block4_conv3").output
        block5_conv3 = vgg16.get_layer("block5_conv3").output 
        out_vgg16 = vgg16.get_layer("block5_pool").output
  
        #--mid convolutions--
        convMid_1 = self.double_conv2D(n_filters*16, 3, out_vgg16 , batch_norm=batch_Norm, dropout=drop_out)    

        print(convMid_1.get_shape()[1])
        print(type(convMid_1.get_shape()))
        
#        ------- up path ---------- 
#        upconv_1 = Convolution2DTranspose(n_filters*8, (2, 2), strides=(2, 2), use_bias= USE_BIAS)(convMid_1)
#        upconv_1 = self.SubpixelConv2D((convMid_1.get_shape()[1], convMid_1.get_shape()[2], convMid_1.get_shape()[3]), scale=2, name='subpix1')(convMid_1)  
        upconv_1 = UpSampling2D((2,2), interpolation=interpolation)(convMid_1)
        conca_1 = concatenate([upconv_1, block5_conv3], axis=self.axis)
        conv_1 = self.double_conv2D(n_filters*8, 3, conca_1, batch_norm=batch_Norm, dropout=drop_out)
        conv_1 = self.bn_conv2D(n_filters*8, 3, conv_1)
                
#        upconv_2 = Convolution2DTranspose(n_filters*8, (2, 2), strides=(2, 2), use_bias= USE_BIAS)(conv_1)        
#        upconv_2 = self.SubpixelConv2D((conv_1.get_shape()[1], conv_1.get_shape()[2], conv_1.get_shape()[3]), scale=2, name='subpix2')(conv_1)
        upconv_2 = UpSampling2D((2,2), interpolation=interpolation)(conv_1)
        conca_2 = concatenate([upconv_2, block4_conv3], axis=self.axis)
        conv_2 = self.double_conv2D(n_filters*8, 3, conca_2, batch_norm=batch_Norm, dropout=drop_out)
        conv_2 = self.bn_conv2D(n_filters*8, 3, conv_2)
        
#        upconv_3 = Convolution2DTranspose(n_filters*4, (2, 2), strides=(2, 2), use_bias= USE_BIAS)(conv_2)
#        upconv_3 = self.SubpixelConv2D((conv_2.get_shape()[1], conv_2.get_shape()[2], conv_2.get_shape()[3]), scale=2, name='subpix3')(conv_2)
        upconv_3 = UpSampling2D((2,2), interpolation=interpolation)(conv_2)
        conca_3 = concatenate([upconv_3, block3_conv3], axis=self.axis)
        conv_3 = self.double_conv2D(n_filters*4, 3, conca_3, batch_norm=batch_Norm, dropout=drop_out) 
        conv_3 = self.bn_conv2D(n_filters*4, 3, conv_3)
        
#        upconv_4 = Convolution2DTranspose(n_filters*2, (2, 2), strides=(2, 2), use_bias= USE_BIAS)(conv_3)
#        upconv_4 = self.SubpixelConv2D((conv_3.get_shape()[1], conv_3.get_shape()[2], conv_3.get_shape()[3]), scale=2, name='subpix4')(conv_3)
        upconv_4 = UpSampling2D((2,2), interpolation=interpolation)(conv_3)
        conca_4 = concatenate([upconv_4, block2_conv2], axis=self.axis)
        conv_4 = self.double_conv2D(n_filters*2, 3, conca_4, batch_norm=batch_Norm, dropout=drop_out)        
    
#        upconv_5 = Convolution2DTranspose(n_filters, (2, 2), strides=(2, 2), use_bias= USE_BIAS)(conv_4)
#        upconv_5 = self.SubpixelConv2D((conv_4.get_shape()[1], conv_4.get_shape()[2], conv_4.get_shape()[3]), scale=2, name='subpi5')(conv_4)
        upconv_5 = UpSampling2D((2,2), interpolation=interpolation)(conv_4)
        conca_5 = concatenate([upconv_5, block1_conv2], axis=self.axis)
        conv_5 = self.double_conv2D(n_filters, 3, conca_5, batch_norm=batch_Norm, dropout=drop_out)    
            
        conca_6 = concatenate([conv_5, inputs], axis=self.axis)
        out = Conv2D(output_channels, (1, 1))(conca_6)
        out = Activation('softmax')(out)

        model = Model(input=inputs, output=out, name="unet_vgg16")
        return model 
  

# ===================== Unet with VGG16 encoder ===============================    
    
    def get_VGG16_Unet(self, img_rows=320, img_cols=320, input_channels=1, output_channels=4, kernel_size = 5, drop_out=0.0, batch_Norm=True, interpolation='bilinear'):
        '''
        plain VGG16 architecture without loading weights 
        '''        
        n_filters = 64
        inputs = Input((img_rows, img_cols, 3))
        
#        ----- encoding path ------
        block1_conv2 = self.double_conv2D(n_filters, 3, inputs, batch_norm=batch_Norm, dropout=drop_out)
        pool_1 = MaxPooling2D(pool_size=(2, 2))(block1_conv2)

        block2_conv2 = self.double_conv2D(n_filters*2, 3, pool_1, batch_norm=batch_Norm, dropout=drop_out)
        pool_2 = MaxPooling2D(pool_size=(2, 2))(block2_conv2)
        
        block3_conv2 = self.double_conv2D(n_filters*4, 3, pool_2, batch_norm=batch_Norm, dropout=drop_out)
        block3_conv3 = self.bn_conv2D(n_filters*4, 3, block3_conv2)           
        pool_3 = MaxPooling2D(pool_size=(2, 2))(block3_conv3)
     
        block4_conv2 = self.double_conv2D(n_filters*8, 3, pool_3, batch_norm=batch_Norm, dropout=drop_out)
        block4_conv3 = self.bn_conv2D(n_filters*8, 3, block4_conv2)           
        pool_4 = MaxPooling2D(pool_size=(2, 2))(block4_conv3)
        
        block5_conv2 = self.double_conv2D(n_filters*8, 3, pool_4, batch_norm=batch_Norm, dropout=drop_out)
        block5_conv3 = self.bn_conv2D(n_filters*8, 3, block5_conv2)           
        pool_5 = MaxPooling2D(pool_size=(2, 2))(block5_conv3)
                
         #--mid convolutions--    
        convMid_1 = self.double_conv2D(n_filters*16, 3, pool_5, batch_norm=batch_Norm, dropout=drop_out)    

#       ------- decoder path ----------
        upconv_1 = UpSampling2D((2,2), interpolation=interpolation)(convMid_1)
        conca_1 = concatenate([upconv_1, block5_conv3], axis=self.axis)
        conv_1 = self.double_conv2D(n_filters*8, kernel_size, conca_1, batch_norm=batch_Norm, dropout=drop_out)
        conv_1 = self.bn_conv2D(n_filters*8, 3, conv_1)
                
#        upconv_2 = self.SubpixelConv2D((conv_1.get_shape()[1], conv_1.get_shape()[2], conv_1.get_shape()[3]), scale=2, name='subpix2')(conv_1)
        upconv_2 = UpSampling2D((2,2), interpolation=interpolation)(conv_1)
        conca_2 = concatenate([upconv_2, block4_conv3], axis=self.axis)
        conv_2 = self.double_conv2D(n_filters*8, kernel_size, conca_2, batch_norm=batch_Norm, dropout=drop_out)
        conv_2 = self.bn_conv2D(n_filters*8, 3, conv_2)
        
#        upconv_3 = self.SubpixelConv2D((conv_2.get_shape()[1], conv_2.get_shape()[2], conv_2.get_shape()[3]), scale=2, name='subpix3')(conv_2)
        upconv_3 = UpSampling2D((2,2), interpolation=interpolation)(conv_2)
        conca_3 = concatenate([upconv_3, block3_conv3], axis=self.axis)
        conv_3 = self.double_conv2D(n_filters*4, kernel_size, conca_3, batch_norm=batch_Norm, dropout=drop_out) 
        conv_3 = self.bn_conv2D(n_filters*4, 3, conv_3)
        
#        upconv_4 = self.SubpixelConv2D((conv_3.get_shape()[1], conv_3.get_shape()[2], conv_3.get_shape()[3]), scale=2, name='subpix4')(conv_3)
        upconv_4 = UpSampling2D((2,2), interpolation=interpolation)(conv_3)
        conca_4 = concatenate([upconv_4, block2_conv2], axis=self.axis)
        conv_4 = self.double_conv2D(n_filters*2, kernel_size, conca_4, batch_norm=batch_Norm, dropout=drop_out)        
    
#        upconv_5 = self.SubpixelConv2D((conv_4.get_shape()[1], conv_4.get_shape()[2], conv_4.get_shape()[3]), scale=2, name='subpi5')(conv_4)
        upconv_5 = UpSampling2D((2,2), interpolation=interpolation)(conv_4)
        conca_5 = concatenate([upconv_5, block1_conv2], axis=self.axis)
        conv_5 = self.double_conv2D(n_filters, kernel_size, conca_5, batch_norm=batch_Norm, dropout=drop_out)    
            
        conca_6 = concatenate([conv_5, inputs], axis=self.axis)
        out = Conv2D(output_channels, (1, 1))(conca_6)
        out = Activation('softmax')(out)

        model = Model(input=inputs, output=out, name="unet_vgg16")
        return model 

    
#================ VGG Network to train with weights maps ======================== 
    def get_VGG16_Unet_weight(self, img_rows=320, img_cols=320, input_channels=1, output_channels=4, drop_out=0.0, batch_Norm=True, USE_BIAS = False):
            
            n_filters = 64
            inputs = Input((img_rows, img_cols, 3))
    
            #get VGG16
            vgg16 = VGG16(input_tensor=inputs, include_top=False)
            for l in vgg16.layers:
                l.trainable = True
                
            out_vgg16 = vgg16(inputs)
            #get vgg layer outputs    
            block1_conv2 = vgg16.get_layer("block1_conv2").output    
            block2_conv2 = vgg16.get_layer("block2_conv2").output
            block3_conv3 = vgg16.get_layer("block3_conv3").output      
            block4_conv3 = vgg16.get_layer("block4_conv3").output
            block5_conv3 = vgg16.get_layer("block5_conv3").output 
            out_vgg16 = vgg16.get_layer("block5_pool").output
            
            #--mid convolutions--
            convMid_1 = self.double_conv2D(n_filters*16, 3, out_vgg16 , batch_norm=batch_Norm, dropout=drop_out)    

            #------- up path ---------- 
            upconv_1 = Convolution2DTranspose(n_filters*8, (2, 2), strides=(2, 2), use_bias= USE_BIAS)(convMid_1)
            conca_1 = concatenate([upconv_1, block5_conv3], axis=self.axis)
            conv_1 = self.double_conv2D(n_filters*8, 3, conca_1, batch_norm=batch_Norm, dropout=drop_out)
            conv_1 = self.bn_conv2D(n_filters*8, 3, conv_1)
                    
            upconv_2 = Convolution2DTranspose(n_filters*8, (2, 2), strides=(2, 2), use_bias= USE_BIAS)(conv_1)
            conca_2 = concatenate([upconv_2, block4_conv3], axis=self.axis)
            conv_2 = self.double_conv2D(n_filters*8, 3, conca_2, batch_norm=batch_Norm, dropout=drop_out)
            conv_2 = self.bn_conv2D(n_filters*8, 3, conv_2)
            
            upconv_3 = Convolution2DTranspose(n_filters*4, (2, 2), strides=(2, 2), use_bias= USE_BIAS)(conv_2)
            conca_3 = concatenate([upconv_3, block3_conv3], axis=self.axis)
            conv_3 = self.double_conv2D(n_filters*4, 3, conca_3, batch_norm=batch_Norm, dropout=drop_out) 
            conv_3 = self.bn_conv2D(n_filters*4, 3, conv_3)
            
            upconv_4 = Convolution2DTranspose(n_filters*2, (2, 2), strides=(2, 2), use_bias= USE_BIAS)(conv_3)
            conca_4 = concatenate([upconv_4, block2_conv2], axis=self.axis)
            conv_4 = self.double_conv2D(n_filters*2, 3, conca_4, batch_norm=batch_Norm, dropout=drop_out)        
        
            upconv_5 = Convolution2DTranspose(n_filters, (2, 2), strides=(2, 2), use_bias= USE_BIAS)(conv_4)
            conca_5 = concatenate([upconv_5, block1_conv2], axis=self.axis)
            conv_5 = self.double_conv2D(n_filters, 3, conca_5, batch_norm=batch_Norm, dropout=drop_out)    
            
            out = Conv2D(output_channels, (1, 1))(conv_5)
            softmax_output = Activation('softmax')(out)
            log_activation = softmaxLayer()(softmax_output)
            
            # Add a new input to serve as the source for the weight maps
            weight_map_ip = Input(shape=(img_rows, img_cols, 4))
            weighted_softmax = multiply([log_activation, weight_map_ip])
            model = Model(inputs=[inputs, weight_map_ip], outputs=[weighted_softmax], name='NewUnet')      
            return model 


def createWeightacceptableUnet(modelPath):
    model = load_model(modelPath, custom_objects={ 'class_weighted_pixelwise_crossentropy': class_weighted_pixelwise_crossentropy }) 
    normalize_activation = Lambda(lambda x: x / tf.reduce_sum(x, len(x.get_shape()) - 1, True))(model.output)
    clip_activation = Lambda(lambda x: tf.clip_by_value(x, 10e-8, 1.-10e-8))(normalize_activation)
    log_activation = Lambda(lambda x: K.log(x))(clip_activation)
    # Add a new input to serve as the source for the weight maps
    weight_map_ip = Input(shape=(320, 320, 4), name="input_2")
    weighted_softmax = multiply([log_activation, weight_map_ip])
    model = Model(inputs=[model.input, weight_map_ip], outputs=[weighted_softmax], name='NewUnet')  
    model.summary()    
    return model
    

# def main():
#     um = Unet_Manager()    
#     model = um.get_mynet_2()
#     model.summary()
    
    
# if __name__ == "__main__":
#     main()     
