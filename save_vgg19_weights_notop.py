# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:08:54 2019

@author: David
"""

import keras

model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
model.save_weights("vgg19_weights_notop.h5")
