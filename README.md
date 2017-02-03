## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, my goal is to write a software pipeline to identify the lane boundaries in a video.  

Project Breakdown:
---

1. Camera Calibration  
  * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

2. Pipeline  
  * Apply a distortion correction to raw images.
  * Use color transforms, gradients, etc., to create a thresholded binary image.
  * Apply a perspective transform to rectify binary image ("birds-eye view").
  * Detect lane pixels and fit to find the lane boundary.
  * Determine the curvature of the lane and vehicle position with respect to center.
  * Warp the detected lane boundaries back onto the original image.
  * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

3. Pipeline (video)  

4. Discussion  

1. Camera Calibration: 
---

_Script:_ `myCameraCalibration.py`

To calibrate the camera, I apply the `findChessboradCorners` and `calibrateCamera` functions from `opencv` on 20 9X6-corner chessboard images in `camera_cal`.  **20 is the suggested least number of images to perform a good camera calibration.** I save the found camera calibration matrix and distortion coefficients in a dictionary object and dump it as a pickle file `./output/myCalibration.p`. 

Here is an example of an undistorted chessboard image. 

![image1](./output_images/camera_cal_example_chessboard.png)

2. Pipeline: Undistort
---

_Scripts:_ `myLaneDetection.py` (Line 17) 

This is a straightforward step. Once I load my camera calibration matrix and distortion coefficients (Line 19-21 `./output/myCalibration.p`), I make use of `opencv`'s `undistort` function to undistort a lane image.

Here is an example of an undistorted lane image. 

![image2](./output_images/camera_cal_example_lane.png)

2. Pipeline: Color and Gradient Thresholding
---

_Scripts:_ `myImageProcessing.py` (Line 29-63), `myLaneDetection.py` (Line 20)

I use three sets of thresholding conditions on the original lane color images.  

1. Gradient changes in the x direction between 20 and 100.  
2. S channel values in HLS color channels between 170 and 255.  
3. Yellow and white colors in RGB color channels, where R > 180, G > 180, and B < 155.  

Here are some examples of my binary thresholded images. 

![image3](./output_images/binary_thresholded.png)

2. Pipeline: Perspective Transform/Warping
---

_Scripts:_ `myCameraCalibration.py` (Line 51, 53, 56), `myImageProcessing.py` (Line 67-71), `myLaneDetection.py` (Line 23)

I handpicked four source points that roughly formed an isosceles trapezoid in the original 3D lane image, labeled with pink crosses below, and defined their destination points on a 2D image plane.   

src | dst |
---|---|
(595,450)|(400,0)|
(689,450)|(880,0)|
(1060,690)|(880,720)|
(250,690)|(400,720)|

![image4](./output_images/perspective_transform_src.png)

I confirmed my choices for the source and destination points after all the warped lane images showed somewhat satisfactory parallel lane lines.  

![image5](./output_images/perspective_transformed_and_binary_thresholded.png)

2. Pipeline: Slidng Window Method
---

_Scripts:_ `myImageProcessing.py` (Line 75-200), `myLaneDetection.py` (Line 26)

This is probably the part of the project where I have spent most time on because it is challenging to always detect lane lines correctly. I rely on two functions to do my job properly. They are `find_lane_start` and `sliding_window_method`. Here is roughly how they work. Detailed comments can be found in the script.  

The warped image from the previous step gets passed into my `sliding_window_method` function with the default sliding window size set to be (128, 72). If both the left and right lane lines are detected in the previous frame, searching of lanes in the current frame starts where the lanes in the previous frame start. _Side note: alternatively, I can create a region of interest using the lane information from the previous frame or frames._  In all other cases, if only one lane was detected and not the other in the previous frame, I set the starting point of my search for the detected lane where the previously found lane is, and use a custom function to locate a good starting point for my search; if neither lanes were detected in the previous frame, a full-blown search gets kicked off.  

I will explain `find_lane_start` next before I come back to `sliding_window_method`. `find_lane_start` gets passed the histogram information of the lower half of a warped lane image. It first tries to look for peaks that have a value greater than 4,500 pixels and a wavelength of (50, 100) pixels on the left and right panels. The latter is done using `scipy.signals.find_peaks_cwt`. If those peaks are found, I then assume that that true lane lines are the ones that are closer to the center line. If those peaks are not found, perhaps my criteria are too strigent for certain cases, and I will resort to simply finding the x-coordinates of the pinnacles in the histogram.  

With information of where to start the search for the lane lines, I can draw my very first two sliding windows. In cases where the lanes are present in the first two sliding windows, the search moves upward, adjusting their positions based on where there is a higher concentration of pixels. However, there may still be cases where nothing is detected in the sliding windows, or they are in the wrong positions. For now, I can let that be. 

During the search, I use standard deviation - which is tied to variance - to decide if what I see in a sliding window gives me good intel about the presence of a lane line. As I have noticed, points can spread every which way in certain sliding windows. This would force `signal.find_peaks_cwt` to analyze a flat line, resulting in very inaccurate suggestions of lane lines. Since I only feel confident to shift my sliding windows left and right when there is clear indication of a clean lane line, I set a condition to the standard deviation value of the pixels in the sliding window. Noisy dots results in very small variance/standard deviation, and clear lane lines results in very large variance/stardard deviation. Through trial and error, I found that 1000 for stardard deviation worked very well.  

Here are some examples of my detected lane lines.  

![image6](./output_images/slidingwindowed.png)

2. Pipeline: Find Curvature and Position of Car Off-center
---

2. Pipeline: Warping
---

2. Pipeline: Assemble Diagnostic Screen
---

3. Pipeline: Video
---

4. Discussion
---

 
