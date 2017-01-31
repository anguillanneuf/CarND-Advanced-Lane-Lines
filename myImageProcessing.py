#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:09:07 2017

@author: tz
"""

import pickle
import cv2
import numpy as np
from scipy import signal

# load camera calibration matrix and distortion coefficients.   
myCalibration = pickle.load(open("./output/myCalibration.p", "rb"))

mtx, dist = myCalibration['mtx'], myCalibration['dist']
M, Minv = myCalibration['M'], myCalibration['Minv']

h,w,c = 720, 1280, 3

y_arr = np.linspace(0, h, num = h+1) # array([0., 1., 2., ..., 720.])


def thresholding(img, hls_thresh=(170,255), gradx_thresh=(20,100)):
    '''
    `img`: raw image of RGB color channels.
    
    '''
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Sobel X takes gradients along the x-axis and emphasizes vertical lines. 
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    # abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    
    # Scale the gradients to (0,255)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    # Convert RGB to HLS and focus on the S channel. 
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    binary_output = np.zeros_like(gray)
    
    # Three sets of thresholding conditions that consider: 
    # 1. Gradient changes in x-direction.
    # 2. S channel in HLS. 
    # 3. Yellow and white in RGB.

                    
    binary_output[(scaled_sobelx >= gradx_thresh[0]) & 
                   (scaled_sobelx <= gradx_thresh[1])|
                  (s_channel > hls_thresh[0]) & (s_channel <= hls_thresh[1]) &
                  ((img[:,:,0]>180) & (img[:,:,1]>180) & (img[:,:,2]<155))] = 1
                    
    # Output `binary_output` is a black and white image with relatively 
    # distinct lane lines. 
    return binary_output*255
    
    
    
def warping(img):
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    
    # Returns a bird's eye view image and the perspective transform matrices.
    return warped
    

    
def find_lane_start(histogram):
    '''
    `histogram`: an 1-D array that keeps track of the number of pixels with 
                 a value of 1 along the y-axis. 
    '''
    mid_left_start = np.argmax(histogram[:int(w/2)])
    mid_right_start = np.argmax(histogram[int(w/2):])+int(w/2)
    
    # Returns the x-coordinates of the lanes at the bottom of an image. 
    return mid_left_start, mid_right_start

    
    
def sliding_window_method(warped, δh=64, δv=72):
    '''
    `warped`: perspective transformed image.
    `δh`: the width of the sliding window divided by 2.
    `δv`: the height of the sliding window.
    
    '''
    
    histogram = np.sum(warped[int(warped.shape[0]/2):, :], axis=0)
    
    # Find the centers of the first two sliding windows. 
    mlx, mrx = find_lane_start(histogram)
    
    # Let y be 720. As the sliding windows travel upward, y decreases. 
    y = h
    
    # Create a blank image. 
    lane_pts = np.zeros_like(warped)

    while y > 0:
        # Zoom into the left and right sliding windows. 
        bbox_left = warped[(y-δv):y, np.clip((mlx-δh),0,1280):np.clip((mlx+δh),0,1280)]
        bbox_right = warped[(y-δv):y, np.clip((mrx-δh),0,1280):np.clip((mrx+δh),0,1280)]
        
        
        # Update `lane_pts` based on sliding windows, where pixel values=255.  
        lane_pts[(y-δv):y, np.clip((mlx-δh),0,1280):np.clip((mlx+δh),0,1280)][(bbox_left==255)] = 1
        lane_pts[(y-δv):y, np.clip((mrx-δh),0,1280):np.clip((mrx+δh),0,1280)][(bbox_right==255)] = 1
        
        # Use new histogram to find lane lines, where there is the highest
        # concentration of pixels. 
        hist_left = np.sum(bbox_left, axis=0)
        hist_right = np.sum(bbox_right, axis=0)
        
        # Define peaks to have a width between 10 and 30 pixels. 
        peakind_left = signal.find_peaks_cwt(hist_left, np.arange(10,30))
        peakind_right = signal.find_peaks_cwt(hist_right, np.arange(10,30))

        
        # If peaks are found, update sliding window centers. 
        if len(peakind_left)>0:
            mlx = int(np.clip(peakind_left[0]+mlx-δh,0,w/2))

        if len(peakind_right)>0:
            mrx = int(np.clip(peakind_right[0]+mrx-δh,w/2,w))

        # Shift sliding window upward until it hits the top of the image. 
        y -= δv
        
    return lane_pts
    

# Calculate R squared of a fit.
def calcR2(x, y, coeff):
    # Construct the polynomial. 
    p = np.poly1d(coeff)
    
    yhat = p(x)
    ybar = np.mean(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
     
    return 1.-ssreg / sstot
    

# Calculate the fitted x values.
def calcFitx(y_arr, fit):
    return fit[0]*y_arr**2 + fit[1]*y_arr + fit[2]
    

# Calculate curvature.  
def calcCurv(v, fit):
    return ( (1+(2*fit[0]*v+fit[1])**2)**1.5 ) / np.absolute(2*fit[0])
  


def findCurvature(lane_slidingwindowed, y_arr=y_arr):
    '''
    `lane_slidingwindowed`: this is the most clean version of the bird's eye 
                            view of the lane lines. 
                            
    '''
    
    # Gather the x and y coordinates of the pixels belonging to the lane lines.
    xvals = np.where(lane_slidingwindowed == 1)[1]
    yvals = np.where(lane_slidingwindowed == 1)[0]

    
    # Group the coordinates into left and right lanes. 
    leftx,lefty = xvals[xvals<=w/2],yvals[xvals<=w/2]
    rightx,righty = xvals[xvals>=w/2],yvals[xvals>=w/2]
    

    # Fit 2 degree polynomials, using y values as x inputs, as vice versa.
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = calcFitx(y_arr, left_fit)

    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = calcFitx(y_arr, right_fit)

    # Use suggested meters-per-pixel conversion. 
    ym_per_pix = 30/h
    xm_per_pix = 3.7/(w*(6/16))

    # Fit 2 degree polynomials on converted x and y values. 
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Find out the y-coordinate for the pixels closest to the camera. 
    lefty_max, righty_max = np.max(lefty*ym_per_pix), np.max(righty*ym_per_pix)
    
    # Calculate road curvature in meters.    
    left_curverad = calcCurv(lefty_max, left_fit_cr)
    right_curverad = calcCurv(righty_max, right_fit_cr)
    
    # Calculate R squared for the fit. 
    left_fit_r2 = calcR2(lefty, leftx, left_fit)
    right_fit_r2 = calcR2(righty, rightx, right_fit)
    
    # Determine the distance that the car is off center in meters.
    offcenter = ((right_fitx[0]+left_fitx[0])/2-w/2)*(3.7/(w*(6/16)))
    
    return (leftx, rightx, lefty, righty, left_fitx, right_fitx, left_curverad,
            right_curverad, left_fit_r2, right_fit_r2, offcenter)


    
def drawCurves(lx, rx, ly, ry, lfx, rfx):
    lane_detected = np.zeros((h,w,c))
    
    lane_detected[:,:,0][ly, lx] = 255
    
    lane_detected[:,:,1][np.hstack([y_arr-1]*10).astype(int), \
    np.clip(np.hstack([lfx-2,lfx-1,lfx,lfx+1,lfx+2,rfx-2,rfx-1,rfx,rfx+1,rfx+2]).astype(int),0,1279)] = 255
    
    lane_detected[:,:,2][ry, rx] = 255
    
    return lane_detected


def unwarping(lane_dst, lane_slidingwindowed):
    # Find curvature.
    lx,rx,ly,ry,lfx,rfx,lc,rc,lr2,rr2,oc = findCurvature(lane_slidingwindowed)
    
    warp_zero = np.zeros_like(lane_slidingwindowed)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Draw a polygon using the fitted x and y values.
    pts_left = np.array([np.transpose(np.vstack([lfx,y_arr]))])
    pts_right = np.array([np.transpose(np.vstack([rfx[::-1],y_arr[::-1]]))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]),(0,255,0))

    # Unwarp and overlay. 
    lane_unwarped = cv2.warpPerspective(color_warp, Minv, (w,h))
    lane_overlayed = cv2.addWeighted(lane_dst, 1, lane_unwarped, 0.3, 0)
    
    return lane_overlayed
    
    
def createDiagScreen(diag1, diag2, diag3, diag4, diag5, info):
    font = cv2.FONT_HERSHEY_COMPLEX
    textpanel = np.zeros((120,1280,3),dtype=np.uint8)
    
    curvrad = np.mean([info['lc'], info['rc']])
    mytext = "Estimated lane curvature: {:.2f}\
    Estimated Meters left of center: {:.2f}\
    R-squared left: {:.2f}\
    R-squared right: {:.2f}".\
    format(curvrad, info['oc'], info['lr2'], info['rr2'])
              
    cv2.putText(textpanel, mytext, (30,60), font, 0.5, (255,255,255), 1)
    
    diagScreen = np.zeros((840,1680,3), dtype=np.uint8)
    diagScreen[0:720,0:1280] = diag1
    diagScreen[720:840,0:1280] = textpanel
    diagScreen[0:210,1280:1680] = cv2.resize(cv2.cvtColor(diag2, cv2.COLOR_GRAY2BGR), (400,210), interpolation=cv2.INTER_AREA)
    diagScreen[210:420,1280:1680] = cv2.resize(cv2.cvtColor(diag3, cv2.COLOR_GRAY2BGR), (400,210), interpolation=cv2.INTER_AREA)
    diagScreen[420:630,1280:1680] = cv2.resize(cv2.cvtColor(diag4, cv2.COLOR_GRAY2BGR), (400,210), interpolation=cv2.INTER_AREA)
    diagScreen[630:840,1280:1680] = cv2.resize(diag5, (400,210), interpolation=cv2.INTER_AREA)

    return diagScreen