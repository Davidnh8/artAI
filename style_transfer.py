# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:35:06 2019

@author: David
"""

import Artist
import numpy as np

content="images//scream.jpg"
style="images//picasso.jpg"
iteration_count=30

save_path=content.split('.')[0][8:] + "-" +style.split('.')[0][8:] + "-iter=" + str(iteration_count) + ".jpg"
#save_path="dont_override.jpg"

artist = Artist.Artist(content,
                style,
                content_coeff=1.0,
                style_coeff=6000.0,
                content_layer_names=['block4_conv2'],
                iteration_count=iteration_count,
                save_path=save_path)