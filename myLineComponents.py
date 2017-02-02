#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:05:14 2017

@author: tz
"""

import numpy as np
from collections import deque

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        
        # x values of the last n fits of the line
        self.fx = deque(maxlen=5)
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = deque(maxlen=5)
        
        #polynomial coefficients for the most recent fit
        self.coeffs = [np.array([False])]  
        #r squared of the best fit
        self.r2 = 1.0 
        
        #radius of curvature of the line in some units
        self.c = None 
        #distance in meters of vehicle center from the line
        self.oc = None 

        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
