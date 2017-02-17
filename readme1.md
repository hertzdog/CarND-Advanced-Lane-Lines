readme.md


Review: the final movie file can be found here: https://youtu.be/dSwJ_oc49og



The goals / steps of this project are the following:

## Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

I first check if calibration has been already done, by checking the calibration file.

If the file is not found, calibration is done using provided images, using the same code used for the project doing calibration of the camera of my [smartphone] (https://github.com/hertzdog/CarND-Camera-Calibration).

The calibration is saved (output_images/camera_cal_pickle.p), ready to be used in the following (also an image is produced output_images/undistort_output.png)


## Apply a distortion correction to raw images.
If the calibration was not made, an image from video is provided (usgin the function extract_frame() and taking a frame know to be straight road).

On that image, distortion correction is performed by the function distortion_correction() which takes as input the file with the calibration and the image to correct (the function will be used also on video)

##Apply a perspective transform to rectify binary image ("birds-eye view").
On the undistorted image pixels were identified for transforming in "birds-eye view".
The SRC pixels where identified as well as the pixels we sould like in the transformed image.
The funtcion is called warp_frame() and follows instruction given during the lessons.
The only "creative" thing done was defying source and destination pixels:
```
offset = 350
# FOUR SOURCE COORDINATES
  src = np.float32(
      [[328, 670],
      [1081, 670],
      [601, 450],
      [687, 450]])

# better
  dst = np.float32(
      [
      [offset, img.shape[0]],
      [img.shape[1]-offset, img.shape[0]],
      [offset, 0],
      [img.shape[1]-offset, 0]
      ])
```


## Use color transforms, gradients, etc., to create a thresholded binary image.

A lot of tests were done on image colors and trasformation. HLS, YUV and GRAY spaces where used to find an averaged final image.
I think still the result can be improved. The code is in the main section of the file: a function has not been done, because I would like to continue testing even after subission, for improving the treatment.

For the thresholds and methods, I folloed what was in the course. I definend:
dir_threshold(), mag_thresh(), abs_sobel_thresh() and combined in a unique image:
```
combined_gray[((gradx_gray == 1)) | ((mag_binary_gray == 1) & (dir_binary_gray == 1))] = 1
```
As was explained in the course.

The result was saved in the image "treatment_on_single_frame.png" as sublots 1 and 2


## Detect lane pixels and fit to find the lane boundary.

Lane pixels were detected with the method explained in the course (histogram).
The method has 2 different fucntions:
find_curvature_init(image)
find_curvature(image, left_fit, right_fit)

The first one is used for finding pixels if no previous persistent fit have been done (for example at the beginnin, or after 5 frames not good).

The other method takes also the previous prediction in order to be able to focus on "proximus" pixels (since the lane should not change too much)


## Determine the curvature of the lane and vehicle position with respect to center.
The curvature is found with fitting a second order polynomial to each lines.
```
left_fit = np.polyfit(lefty, leftx, 2)
```

The code is taken from the course material.

The result is saved on the same figure for the still frame: "output_images/treatment_on_single_frame.png"

The curvature radius has been computed taking into account that:
ym_per_pix = 30/720 # meters per pixel in y dimension (full image height)
xm_per_pix = 3.7/753 # meters per pixel in x dimension (measured from warped image)

The function for curvature is: curvature_radius(ploty, leftx, rightx)

Being the center of the car always in the same position with respect to the camera (they are fixed), being the right/Left offeset of the warped image the same (350 pixels) we measured the difference between the center of the image and the center of the lanes in the first 50 pixels (last 50 values of the leftx and lefty)


## Warp the detected lane boundaries back onto the original image.

For warping the detected lane boundaries back onto the original image we used the inverse trasformation done during warping, thus using the same SRC and DST pixels but in inverted order.
We defined the fucntion:
```
def project_back(warped, ploty, leftx, rightx, undistorted_image)
```
It takes the warped image with the lines and produce a new image stacked on the undistorded image and returns the stacked image.  

## Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

This has been done putting the text on the image using cv2 putText.

```
cv2.putText(last,average_curvature,(10,50), font, 1,(255,255,255),2)
cv2.putText(last,deviation_from_center,(10,80), font, 1,(255,255,255),2)
```


## Sanity Check

For the project was required a sanity check. It has been done on:
```
delta_curvature = 0.2 #20%
delta_distance = 0.1 #10%
delta_parallel = 100 #pixels
```

If the difference in curvature is more than 20% OR the difference in distance among x (last 50px near the car) betweein left and right lines is more than 10% OR the difference the mean point near the car (on 50px) and the farest pint (50px) is more than 100 pixels on X then the lines are not well detected and previous computation will be used.

If it happens more then 5 times a re-initiaziazion is made.
