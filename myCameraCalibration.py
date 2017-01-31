#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:24:34 2017

@author: tz
"""

import cv2
import numpy as np
import glob
import pickle
import os

def main(): 
    
    # dimensions of chessboard corners.
    nx, ny = 9, 6
    
    # prepare object points (0,0,0), (1,0,0), (2,0,0) .. (8,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[:nx, :ny].T.reshape(-1,2)
    
    # arrays to store object points and image points.
    objpoints = [] # 3d points in the real world
    imgpoints = [] # 2d points in image space 
    
    images = glob.glob('camera_cal/*.jpg')
    
    
    # use all available images to calibrate the camera.
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # find chessboard corners.
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        # if found, populate objpoints and imgpoints. 
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)    
    
    w,h = img.shape[1], img.shape[0]
    
    # compute camera calibration matrix and distortion coefficients.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                       (w,h), None, None)
 
    
    src = np.float32([[595,450],[689,450],[1060,690],[250,690]])
    dst = np.float32([[w*5/16,0],[w*11/16,0],[w*11/16,h],[w*5/16,h]])
        
    # M is the perspective transform matrix; Minv is the inversed M. 
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    
    # save camera calibration matrix and distortion coefficients.
    myCalibration = {'mtx': mtx, 'dist': dist, 'M': M, 'Minv': Minv}


    if not os.path.exists('./output'):
        os.makedirs('./output')
    
    pickle.dump(myCalibration, open("./output/myCalibration.p", "wb"))
    
    
if __name__ == '__main__':
    main()