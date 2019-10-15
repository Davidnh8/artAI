# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:35:06 2019

@author: David
"""

import Artist
import numpy as np

artist = Artist.Artist("images/vangogh.jpg",
                "images/picasso.jpg",
                content_coeff=1.0,
                style_coeff=6000.0,
                iteration_count=30,
                save_path="test.jpg")