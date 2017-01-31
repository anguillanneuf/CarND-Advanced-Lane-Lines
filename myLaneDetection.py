#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:16:40 2017

@author: tz
"""
# import myLineComponents as C
import myImageProcessing as I

import cv2
import numpy as np

from moviepy.editor import VideoFileClip



def process_image(lane_img):
    
    lane_dst = cv2.undistort(lane_img, I.mtx, I.dist, None, I.mtx)
    lane_thresholded = I.thresholding(lane_dst)
    lane_warped = I.warping(lane_thresholded)
    lane_slidingwindowed = I.sliding_window_method(lane_warped)
    
    lfx,rfx,lc,rc,lr2,rr2,oc= I.findingCurvature(lane_slidingwindowed)
    
    warp_zero = np.zeros_like(lane_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([lfx,I.y_arr]))])
    pts_right = np.array([np.transpose(np.vstack([rfx[::-1],I.y_arr[::-1]]))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]),(0,255,0))

    lane_unwarped = I.unwarping(lane_dst, lane_slidingwindowed)

    return lane_unwarped
    

project_video_output_fname = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
project_video_output = clip1.fl_image(process_image)
project_video_output.write_videofile(project_video_output_fname, audio=False)



## middle panel text example
#    # using cv2 for drawing text in diagnostic pipeline.
#    font = cv2.FONT_HERSHEY_COMPLEX
#    middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
#    cv2.putText(middlepanel, 'Estimated lane curvature: ERROR!', (30, 60), font, 1, (255,0,0), 2)
#    cv2.putText(middlepanel, 'Estimated Meters right of center: ERROR!', (30, 90), font, 1, (255,0,0), 2)
#
#
#    # assemble the screen example
#    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
#    diagScreen[0:720, 0:1280] = mainDiagScreen
#    diagScreen[0:240, 1280:1600] = cv2.resize(diag1, (320,240), interpolation=cv2.INTER_AREA) 
#    diagScreen[0:240, 1600:1920] = cv2.resize(diag2, (320,240), interpolation=cv2.INTER_AREA)
#    diagScreen[240:480, 1280:1600] = cv2.resize(diag3, (320,240), interpolation=cv2.INTER_AREA)
#    diagScreen[240:480, 1600:1920] = cv2.resize(diag4, (320,240), interpolation=cv2.INTER_AREA)*4
#    diagScreen[600:1080, 1280:1920] = cv2.resize(diag7, (640,480), interpolation=cv2.INTER_AREA)*4
#    diagScreen[720:840, 0:1280] = middlepanel
#    diagScreen[840:1080, 0:320] = cv2.resize(diag5, (320,240), interpolation=cv2.INTER_AREA)
#    diagScreen[840:1080, 320:640] = cv2.resize(diag6, (320,240), interpolation=cv2.INTER_AREA)
#    diagScreen[840:1080, 640:960] = cv2.resize(diag9, (320,240), interpolation=cv2.INTER_AREA)
#    diagScreen[840:1080, 960:1280] = cv2.resize(diag8, (320,240), interpolation=cv2.INTER_AREA)
