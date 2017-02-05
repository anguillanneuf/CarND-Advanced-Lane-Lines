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
import myLineComponents as Line

L = Line.Line()
R = Line.Line()

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
    return binary_output
    
    
def warping(img):
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    
    # Returns a bird's eye view image and the perspective transform matrices.
    return warped
    

    
def find_lane_start(histogram, which = 'both'):
    '''
    `histogram`: an 1-D array that keeps track of the number of pixels with 
                 a value of 1 along the y-axis. 
    '''
    
    left_peaks = signal.find_peaks_cwt(histogram[:int(w/2)], np.arange(50,100))
    # Valid if more than 4,500 pixels are found in a pixel column
    valid_left_peaks = [x for x in left_peaks if histogram[x]>4500]
    
    if len(valid_left_peaks) >0:
        mid_left_start = valid_left_peaks[-1]
    else:
        mid_left_start = np.clip(np.argmax(histogram[:int(w/2)]),0,639)
    
        
        
    right_peaks = signal.find_peaks_cwt(histogram[int(w/2):], np.arange(50,100))
    valid_right_peaks = [x for x in right_peaks if histogram[x+int(w/2)]>4500]
    
    if len(valid_right_peaks) >0:
        mid_right_start = valid_right_peaks[0]+int(w/2)
    else:
        mid_right_start = np.clip(np.argmax(histogram[int(w/2):])+int(w/2),640,1279)
    
        
    # Returns the x-coordinates of the lanes at the bottom of an image.
    if which == 'left': 
        return mid_left_start
    elif which == 'right':
        return mid_right_start
    else:
        return mid_left_start, mid_right_start

def update_mx_from_histogram(bbox=None, mx=None, δh=64, d='left', y=None):
    global L
    global R
    
    # Create histograms for the left and right bounding box. 
    hist = np.sum(bbox, axis=0)

    # Define peaks to have a width between 50 and 100 pixels. Check if 
    # histograms have enough variance in them first. Low variance means 
    # that the histogram is likely uniformly distributed, and the pixels
    # are spread out in the columns, and they are noisy. 

    # np.std(np.array([3,3,3,3,3,3,3,3,3,3])) = 0.0
    # np.std(np.array([0,27,0,0,0,0,0,0,0,0])) = 8.1
    # print("hist var: {}".format(np.var(hist)))
    
    if np.var(hist) > 10: 
        xvals = np.where(bbox == 1)[1]
        yvals = np.where(bbox == 1)[0]
    
        if np.sum(bbox==1)>10:
            fit = np.polyfit(yvals, xvals, 2)
            print("fit[2]: {}".format(fit[2]))
            if d=='left':
                mx = int(np.clip(mx-δh+fit[2], 0, w/2-1))
            else:
                mx = int(np.clip(mx-δh+fit[2], w/2, w-1))
            print(mx)
        else:
            if d=='left':
                mx = L.fx[-1][y]
            else: 
                mx = R.fx[-1][y]
    elif len(L.fx) > 0:
        mx = L.fx[-1][y] if d=='left' else R.fx[-1][y]
      
    return int(mx)
    
    
def sliding_window_method(warped, δh=64, δv=72):
    '''
    `warped`: perspective transformed image.
    `δh`: the width of the sliding window divided by 2.
    `δv`: the height of the sliding window.
    
    '''
    
    # Global variables of class Line. 
    global L
    global R
    
    # STEP 1: Find where to start the search. Return `mlx`, `mrx`. 
    
    # If both left and right lanes are detected. 
    if np.logical_and(L.detected, R.detected):
        # Use fitted x from the previous frame. 
        mlx, mrx = L.fx[-1][-1].astype(int), R.fx[-1][-1].astype(int)
    
    # All other cases. 
    else: 
        histogram = np.sum(warped[int(warped.shape[0]/2):, :], axis=0)
        
        # If either the left or the right lane is detected but not both.
        if np.logical_xor(L.detected, R.detected):
        
            if L.detected:
                mlx = L.fx[-1][-1].astype(int)
                mrx = find_lane_start(histogram, 'right')
            else: 
                mrx = R.fx[-1][-1].astype(int)
                mlx = find_lane_start(histogram, 'left')

        # If neither the left nor the right lanes are detected. 
        else: 
            # Find the centers of the first two sliding windows. 
            mlx, mrx = find_lane_start(histogram, 'both')        
    
    # STEP 2: Search Upward. Update `mlx` and `mrx`. 
    
    # Let y be 720. As sliding windows travel upward, y decreases. 
    y = h
    
    # Create a blank image. 
    lane_pts = np.zeros_like(warped)

    while y > 0:
        # Zoom into the left and right sliding windows. 
        # Clip them using the left, center, and right vertical lines. 
        bbox_left = warped[(y-δv):y, np.clip((mlx-δh),0,639):np.clip((mlx+δh),0,639)]
        bbox_right = warped[(y-δv):y, np.clip((mrx-δh),640,1279):np.clip((mrx+δh),640,1279)]
        
        # Update `lane_pts` based on sliding windows, where pixel values>0.  
        lane_pts[(y-δv):y, np.clip((mlx-δh),0,639):np.clip((mlx+δh),0,639)][(bbox_left>0)] = 1
        lane_pts[(y-δv):y, np.clip((mrx-δh),640,1279):np.clip((mrx+δh),640,1279)][(bbox_right>0)] = 1
        
        # Shift sliding window upward until it hits the top of the image. 
        y -= δv
               
        mlx = update_mx_from_histogram(bbox_left, mlx, δh, 'left', y)
        mrx = update_mx_from_histogram(bbox_right, mrx, δh, 'right', y)
        
    return lane_pts
    

def maskingMethod(lane_img):
    return lane_pts
    
    
# Calculate R squared of a fit.
def calcR2(x, y, coeff):
    # Construct the polynomial. 
    p = np.poly1d(coeff)
    
    yhat = p(x)
    ybar = np.mean(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
     
    return ssreg / sstot
    

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
    global L
    global R
    
    # Gather the x and y coordinates of the pixels belonging to the lane lines.
    xvals = np.where(lane_slidingwindowed == 1)[1]
    yvals = np.where(lane_slidingwindowed == 1)[0]

    
    # STEP 1: update L.allx and R.allx
    
    # Group the coordinates into left and right lanes. 
    leftx,lefty = xvals[xvals<=w/2],yvals[xvals<=w/2]
    rightx,righty = xvals[xvals>w/2],yvals[xvals>w/2]
    
    # If there are too few points to fit a polynomial, use the last fitted x 
    if len(xvals<=w/2)<10:
        leftx=np.mean(np.array(L.fx), axis=0)#L.fx[-1]
        lefty= y_arr
        
    if len(xvals>w/2)<10:
        rightx = np.mean(np.array(R.fx), axis=0)#R.fx[-1]
        righty = y_arr
    
    L.allx = leftx; L.ally = lefty
    R.allx = rightx; R.ally = righty
    
    
    # STEP 2: update L.fx, R.fx, L.detected, R.detected, L.bestx, R.bestx,
    # L.coeffs, R.coeffs, L.best_fit, R.best_fit
    
    # Fit 2 degree polynomials, using y values as x inputs, as vice versa.
    left_fit = np.polyfit(L.ally, L.allx, 2)
    left_fitx = calcFitx(y_arr, left_fit)

    right_fit = np.polyfit(R.ally, R.allx, 2)
    right_fitx = calcFitx(y_arr, right_fit)
    

    # If the distance between the left and right lane are consistent
    # and the distance makes sense. 
    if(np.std(abs(left_fitx - right_fitx))) < 800 and \
      (np.mean(abs(left_fitx - right_fitx))) > w*(5.5/16) and \
      (np.mean(abs(left_fitx - right_fitx))) < w*(6.5/16):
          
        L.fx.append(left_fitx)
        R.fx.append(right_fitx)
        L.detected = True
        R.detected = True
        L.coeffs = left_fit
        R.coeffs = right_fit
        L.best_fit.append(L.coeffs)
        R.best_fit.append(R.coeffs)
        
    # If the distance is not consistent, must decide which lane is better. 
    else: 
        # Determine the width of the road from the previous five frames. 
        # L.fx is a deque list. 
        left_bestx = np.mean(np.array(L.fx), axis=0)
        right_bestx = np.mean(np.array(R.fx), axis=0)
        w_road = (left_bestx - right_bestx)
        
        # If the left lane is a better fit. 
        # diagnosis
        # print(np.std(abs(left_bestx - left_fitx)))
        # print(np.std(abs(right_bestx - right_fitx)))

        if np.std(abs(left_bestx - left_fitx)) < np.std(abs(right_bestx - right_fitx)):
            L.fx.append(left_fitx)
            L.detected = True
            L.coeffs = left_fit
            L.best_fit.append(L.coeffs)
            
            
            # despite everything, if the fit found is a good fit, use it. 
            if calcR2(righty, rightx, right_fit) < 1:
                R.fx.append(left_fitx - w_road)
                rightx = R.fx[-1]; righty = y_arr
                R.detected = False
            else: 
                R.fx.append(right_fitx)
                R.detected = True


        else: 
            R.fx.append(right_fitx)
            R.detected = True
            R.coeffs = right_fit
            R.best_fit.append(R.coeffs)
            
             
            if calcR2(lefty, leftx, left_fit) < 1:
                L.fx.append(right_fitx + w_road)
                leftx = L.fx[-1]; lefty = y_arr
                L.detected = False
            else: 
                L.fx.append(left_fitx)
                L.detected = True
            
        

    # STEP 3: update L.r2, R.r2, L.c, R.c, L.oc, R.oc
    
    # Use suggested meters-per-pixel conversion. 
    ym_per_pix = 30/h
    xm_per_pix = 3.7/(w*(6/16))

    # Fit 2 degree polynomials on converted x and y values. 
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Find out the y-coordinate for the pixels closest to the camera. 
    lefty_max, righty_max = np.max(lefty*ym_per_pix), np.max(righty*ym_per_pix)
    
    # Calculate road curvature in meters.    
    L.c = calcCurv(lefty_max, left_fit_cr)
    R.c = calcCurv(righty_max, right_fit_cr)
    
    # Calculate R squared for the fit. 
    L.r2 = calcR2(lefty, leftx, left_fit)
    R.r2 = calcR2(righty, rightx, right_fit)
    
    # Determine the distance that the car is off center in meters.
    offcenter = ((R.fx[-1][-1]+L.fx[-1][-1])/2-w/2)*(3.7/(w*(6/16)))
    L.oc = offcenter
    R.oc = offcenter
    
    lfx = np.mean(np.array(L.fx), axis=0)
    rfx = np.mean(np.array(R.fx), axis=0)
    
    return (L.allx, R.allx, L.ally, R.ally, lfx, rfx, L.c, R.c, L.r2, R.r2, L.oc)

    
def drawCurves(lx, rx, ly, ry, lfx, rfx):
    

    
    lane_detected = np.zeros((h,w,c))
    
    lane_detected[:,:,0][np.clip(ly.astype(int),0,719), np.clip(lx.astype(int),0,1279)] = 255
    
    lane_detected[:,:,1][np.hstack([y_arr-1]*10).astype(int), \
    np.clip(np.hstack([lfx-2,lfx-1,lfx,lfx+1,lfx+2,rfx-2,rfx-1,rfx,
                       rfx+1,rfx+2]).astype(int),1,1279)] = 255
    
    lane_detected[:,:,2][np.clip(ry.astype(int),0,719), np.clip(rx.astype(int),0,1279)] = 255
    
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
    font = cv2.FONT_HERSHEY_PLAIN
    textpanel = np.zeros((120,1280,3),dtype=np.uint8)
    
    curvrad = np.mean([info['lc'], info['rc']])
    mytext = "Estimated lane curvature: {:.2f}\
    Estimated Meters left of center: {:.2f}\
    R-squared left: {:.2f}\
    R-squared right: {:.2f}".\
    format(curvrad, info['oc'], info['lr2'], info['rr2'])
              
    cv2.putText(textpanel, mytext, (30,60), font, 1, (255,255,255), 1)
    
    diagScreen = np.zeros((840,1680,3), dtype=np.uint8)
    diagScreen[0:720,0:1280] = diag1
    diagScreen[720:840,0:1280] = textpanel
    diagScreen[0:210,1280:1680] = cv2.resize(cv2.cvtColor(diag2, cv2.COLOR_GRAY2BGR), (400,210), interpolation=cv2.INTER_AREA)
    diagScreen[210:420,1280:1680] = cv2.resize(cv2.cvtColor(diag3, cv2.COLOR_GRAY2BGR), (400,210), interpolation=cv2.INTER_AREA)
    diagScreen[420:630,1280:1680] = cv2.resize(cv2.cvtColor(diag4, cv2.COLOR_GRAY2BGR), (400,210), interpolation=cv2.INTER_AREA)
    diagScreen[630:840,1280:1680] = cv2.resize(diag5, (400,210), interpolation=cv2.INTER_AREA)

    return diagScreen