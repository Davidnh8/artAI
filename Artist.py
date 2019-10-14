# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:09:21 2019

@author: David
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras import backend as K
from tqdm import tqdm
import keras
import tensorflow as tf

from scipy.optimize import fmin_l_bfgs_b

import sys
sys.path.insert(1, 'C:\\Users\\David\\Desktop\\Art_style_transfer\\neural_stylization\\')
#tf.compat.v1.disable_eager_execution()
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import time

#from vgg19 import VGG_19 as customvgg19
from keras.layers import AveragePooling2D, Input
from keras.models import Model

def contentLoss(F: "feature representation of generated image", P: "feature representation of the original content") -> "Content Loss":
    ret = 0.5*K.sum(K.square(F-P))
    return ret

"""
def contentGrad(F: "feature representation of generated image", P: "feature representation of the original content") -> "content Loss Gradient":
    grad=np.empty_like(F)
    diff=F-P
    F_positive_msk=(F>0)
    
    grad[F_positive_msk]=diff[F_positive_msk] # F>0
    grad[np.invert(F_positive_msk)]=0 # F<=0
    
    return grad"""


def gramMatrix(F: "nxm matrix")->"Gram Matrix nxn":
    ret = K.dot( F, K.transpose(F))
    return ret

def styleLossSingleLayer(F: "feature representation of generated image", A_: "gram matrix of Feature representation of the original style") ->" Style Loss of Single Layer":
    Fshape=np.shape(F)
    N=int(Fshape[0])
    M=int(Fshape[1])
    G=gramMatrix(F)
    A=gramMatrix(A_)
    return 0.25 * K.sum(K.square(G-A))/( N**2 * M**2)
"""
def styleLossSingleLayerGrad(F: "feature representation of generated image", A_: "Feature representation of the original style")->"Grad of single layer style loss":
    G=gramMatrix(F)
    A=gramMatrix(A_)
    dEdF=np.empty_like(F)
    diff = G-A
    N,M = F.shape
    F_positive_msk=(F>0)
    dEdF=(1/(N**2*M**2)) * np.transpose(np.matmul(np.transpose(F),diff))
    dEdF[np.invert(F_positive_msk)]=0
    
    return dEdF"""


class Artist():
    def __init__(self,
                 content_img_path,
                 style_img_path,
                 content_coeff=1.0,
                 style_coeff=10000.0,
                 content_layer_names=['block4_conv2'],
                 style_layer_names=["block1_conv1",
                                    "block2_conv1",
                                    "block3_conv1",
                                    "block4_conv1",
                                    "block5_conv1"],
                learning_rate=0.0000001
                 ):
        
        self.content_img_path=content_img_path
        self.style_img_path=style_img_path
        self.content_coeff = content_coeff
        self.style_coeff = style_coeff
        self.content_layer_names = content_layer_names
        self.style_layer_names = style_layer_names
        self.learning_rate = learning_rate
        
        self.style_weights=[1/len(self.style_layer_names) for i in range(len(self.style_layer_names))]
        #self.style_weights=[1. for i in range(len(self.style_layer_names))]
        self.content_img= self.pre_process(np.expand_dims(Image.open(self.content_img_path), axis=0).astype('float32'))
        discard,self.H, self.W, self.C=self.content_img.shape
        self.content = K.constant(self.content_img, dtype='float32')
        self.style_img = self.pre_process(np.expand_dims(Image.open(self.style_img_path).resize((self.W,self.H)), axis=0).astype('float32'))
        self.style = K.constant(self.style_img, dtype='float32')
        self.x = K.placeholder(shape=(1,self.H,self.W,3), dtype='float32')
        
        self.content_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=self.content)
        self.style_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=self.style)
        self.x_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=self.x)
        #self.content_model = customvgg19(include_top=False, input_tensor=self.content, pooling='max')
        #self.style_model = customvgg19(include_top=False, input_tensor=self.style, pooling='max')
        #self.x_model = customvgg19(include_top=False,  input_tensor=self.x, pooling='max')
        self.content_model=self.replace_maxpool_with_averagepool(self.content_model)
        self.style_model=self.replace_maxpool_with_averagepool(self.style_model)
        self.x_model=self.replace_maxpool_with_averagepool(self.x_model)
        #self.input=np.copy(self.content_img)
        self.input = np.random.uniform(0, 1 ,size=self.x.shape)
        #result=self.gradient_descent(self.input, 30 )
        #result = self.post_process(result)
        #raise ValueError("STOPPED")
        self.grad_fn = self.get_grad()
        self.loss=None
        self.grad=None
        
        self.loss_history=[]
        
        """
        result = self.gradient_descent(self.input, iterations=4000)
        result = self.post_process(result)
        
        min1=np.abs(np.min(result))
        max1=np.abs(np.max(result))
        print(np.max(result))
        print(np.min(result))
        
        
        result=np.clip(result[0], 0, 255).astype('uint8')
        
        min1=np.abs(np.min(result))
        max1=np.abs(np.max(result))
        print(np.max(result))
        print(np.min(result))
        
        plt.imshow(result)
        """
        result= self.bfgs(self.input, iterations=5)
        result=np.reshape(result, (1, self.H, self.W, 3))
        result = self.post_process(result)
        
        min1=np.abs(np.min(result))
        max1=np.abs(np.max(result))
        print(np.max(result))
        print(np.min(result))
        
        
        #result=np.clip(255.*(result[0]+min1)/(min1+max1), 0, 255).astype('uint8')
        result = np.clip(result[0], 0, 255).astype('uint8')

        
        fig,ax=plt.subplots(2, figsize=[16,16])
        ax[0].plot(np.log(self.loss_history))
        ax[1].imshow(result)
        
        
        K.clear_session()
        
    def replace_maxpool_with_averagepool(self, model):
        maxpool_names=["block1_pool",
                       "block2_pool",
                       "block3_pool",
                       "block4_pool",
                       "block5_pool"]
        
        # get each of the layers
        count = 1 # count which block's pool we are currently at
        model_layers=[l for l in model.layers]
        x=model.input
        model.summary()
        # iterate over each layer, replacing max pool with average pool
        for e,layer in enumerate(model_layers):
            if e>0:
                if layer.name in maxpool_names:
                    #print("A: ", layer.name)
                    #print("A: ", layer)
                    x = AveragePooling2D((2, 2), strides=(2, 2), name="block"+str(count)+"_pool" )(x)
                    count+=1
                else:
                    #print("B: ", layer.name)
                    #print("B: ", layer)
                    x = layer(x)
                
        new_model = Model(input = model.input, output=x)
        new_model.summary()
        return new_model
    
    
    def pre_process(self, img):
        imagenet_rgb=np.array([123.68, 116.779, 103.939], dtype='float64')
        img[:,:,:,np.array([0,1,2])]-=imagenet_rgb
        return img[:,:,:,::-1]
    
    def post_process(self, img):
        imagenet_rgb=np.array([103.939, 116.779, 123.68], dtype='float64')
        img[:,:,:,np.array([0,1,2])]+=imagenet_rgb
        return img[:,:,:,::-1]
    
    def reshape_into_feature_matrix(self, tensor):
        mshape=np.shape(tensor)
        N=mshape[3]
        M=mshape[1]*mshape[2]
        ret = K.transpose(K.reshape(tensor,(M,N)))
        return ret
    

    
    
    def get_grad(self):
        
        tloss=K.variable(0.0)
        #content loss
        closs=K.variable(0.0)
        #assert len(self.P)==len(self.content_layer_names)
        for content_layer_name in self.content_layer_names:
            print("DFSADFDS", self.content_model.get_layer(content_layer_name))
            P = self.content_model.get_layer(content_layer_name).output
            Fc = self.x_model.get_layer(content_layer_name).output
            closs = closs + contentLoss(Fc, P)
        tloss = tloss + closs
            
        #style loss 
        sloss=K.variable(0.0)
        for style_layer_name, w in zip( self.style_layer_names, self.style_weights):
            A_ = self.reshape_into_feature_matrix( self.style_model.get_layer(style_layer_name).output)
            F  = self.reshape_into_feature_matrix( self.x_model.get_layer(style_layer_name).output)
            sloss = sloss + w * styleLossSingleLayer(F, A_)
        tloss = tloss + self.style_coeff*sloss
        
        ret=K.function([self.x], [tloss, K.gradients(tloss, self.x)])
        
        
        return ret
    
    
    def loss_bfgs(self,x):
        #assert self.loss==None
        #assert self.grad==None
        
        #x = np.reshape(x, (1,224,224,3))
        #x = np.reshape(x, (1,224,224,3))
        
        loss_grad = self.grad_fn([x.reshape(1,self.H,self.W,3)])
        self.grad =  loss_grad[1][0].flatten().astype("float64")
        self.loss = loss_grad[0]#.astype('float64')
        self.loss_history.append(self.loss)
        print("loss value", self.loss)
        return self.loss
    
    
    def grad_bfgs(self, x):
    
            
        return self.grad
    

    


    
    def bfgs(self, layer, iterations):
        layer=layer.flatten().astype('float64')
        for i in tqdm(range(iterations)):
            layer, fmin, info = fmin_l_bfgs_b(self.loss_bfgs, layer, 
                                              fprime=self.grad_bfgs, 
                                              maxfun=20)
        return layer
    
    
    
    
    def gradient_descent(self, layer, iterations):
        
        for i in tqdm(range(iterations)):
            
            loss_grad = self.grad_fn(layer)
            self.grad = loss_grad[1][0]
            self.loss = loss_grad[0]
            #print(self.grad)
            layer -= self.learning_rate*self.grad
            print(self.loss)
        return layer    
            

    
    """
    def gradientDescent(self, learning_rate):
        cont=self.get_feature_maps(self.)
        cont, style=self.get_feature_maps()
        F=[i for i in cont.values()]
        GG= [i for i in style.values()]
        
        contentloss = contentLoss(F, self.P)
        contentgrad = contentGrad(F, self.P)
        total_style_loss=0.
        total_style_grad=0.
        for g,a,w in zip(GG,self.AA, self.style_weights):
            total_style_loss+=w*styleLossSingleLayer(g,a)
            total_style_grad=K.gradients(self.)
        total_loss=self.content_coeff*contentloss + self.style_coeff*total_style_loss
        """
        
artist=Artist("images/vangogh2.jpg", "images/picasso.jpg")

                 
                 




                 
                 
                 
                 
                 
                 