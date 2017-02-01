#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:16:40 2017

@author: tz
"""
# import myLineComponents as L
import myImageProcessing as I
import cv2
import os
# from moviepy.editor import VideoFileClip
# import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob



def process_image(lane_img):
    
    lane_dst = cv2.undistort(lane_img, I.mtx, I.dist, None, I.mtx)
    
    # diag2
    lane_thresholded = I.thresholding(lane_dst)
    
    # diag3
    lane_warped = I.warping(lane_thresholded)
    
    # diag4
    lane_slidingwindowed = I.sliding_window_method(lane_warped)
    
    lx,rx,ly,ry,lfx,rfx,lc,rc,lr2,rr2,oc= I.findCurvature(lane_slidingwindowed)
    info = {'lc':lc, 'rc': rc, 'lr2':lr2, 'rr2': rr2, 'oc': oc}
    
    # diag5
    lane_detected = I.drawCurves(lx, rx, ly, ry, lfx, rfx)

    # diag1 
    lane_unwarped = I.unwarping(lane_dst, lane_slidingwindowed)

    # assemble diagnostic screens
    diagScreen = I.createDiagScreen(lane_unwarped, lane_thresholded, 
                                  lane_warped, lane_slidingwindowed*255,
                                  lane_detected, info)
    return diagScreen
 

def main():
    
#    project_video_output_fname = 'project_video_output.mp4'
#    clip1 = VideoFileClip("project_video.mp4")
#    project_video_output = clip1.fl_image(process_image)
#    project_video_output.write_videofile(project_video_output_fname, 
#                                         audio=False)
    
if __name__ == '__main__':
    main()


