import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os.path
from pathlib import Path
import matplotlib.image as mpimg
from PIL import Image
import moviepy.editor as mpy

#CAMERA CALIBRATION START (Taken from Camera Calibration project)
def camera_calibration():
    """perform camera calibration"""
    # perform camera calibration on given images
    # save the result in camera_cal_pickle.p
    # resize to rows cols
    # normalize
    print ("Camera calibration...")

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')


    #ADDED TO BE ABLE TO CONTROL THE WINDOW
    WINDOW_NAME = 'Calibration'
    cv2.namedWindow(WINDOW_NAME)
    cv2.startWindowThread()

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            #cv2.imshow(WINDOW_NAME, img)
            #cv2.waitKey(1)

            # those are needed on mac: see http://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv
            #cv2.waitKey(1)
            #cv2.destroyAllWindows()
            #cv2.waitKey(1)
            #

            # Do camera calibration given object points and image points
            img_size = (img.shape[1], img.shape[0])
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
            # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
            dist_pickle = {}
            dist_pickle["mtx"] = mtx
            dist_pickle["dist"] = dist
            pickle.dump( dist_pickle, open( "output_images/camera_cal_pickle.p", "wb" ) )
            # END of CAMERA CALIBRATION --------------------------------



def distortion_correction (calibration_file_input, input_img):
    """perform distorion correction of an image, given the calibration file"""

    dist_pickle = {}
    img_size = (input_img.shape[1], input_img.shape[0])
    file = open(calibration_file_input ,'rb')
    dist_pickle = pickle.load(file)
    file.close()
    mtx = dist_pickle["mtx"]
    dist= dist_pickle["dist"]
    undistorted_image = cv2.undistort(input_img, mtx, dist, None, mtx)
    return undistorted_image

def verify_camera_calibration():
    """verify camera calibration"""
    #Distortion correction --------------------------------------
    print ("Testing distortion correction...")
    img = cv2.imread('camera_cal/calibration2.jpg')
    dst = distortion_correction("output_images/camera_cal_pickle.p",img)
    #cv2.imwrite('output_temp/calibration2_undistorted.jpg',dst)
    #print ("Undistorted image saved in ./output_temp")

    # plot with various axes scales
    plt.figure(1)
    #original
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Original')
    #undistorted
    plt.subplot(122)
    plt.imshow(dst)
    plt.title('Undistorted')
    #save
    filename = "output_images/undistort_output.png"
    plt.savefig(filename)

def extract_frame(seconds, video):
    milliseconds = seconds*1000
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC,milliseconds)      # just cue to 20 sec. position
    success,image = cap.read()
    if success:
        return image

def warp_frame(img):
    # we defined points in perspective_transform.py

    offset=350
    # FOUR SOURCE COORDINATES
    src = np.float32(
        [[328, 670],
        [1081, 670],
        [601, 450],
        [687, 450]])

    # FOUR DESTINATION COORDINATES
    dst = np.float32(
        [
        [src[0][0], img.shape[0]],
        [src[1][0], img.shape[0]],
        [src[0][0], 0],
        [src[1][0], 0]
        ])

    # better
    dst = np.float32(
        [
        [offset, img.shape[0]],
        [img.shape[1]-offset, img.shape[0]],
        [offset, 0],
        [img.shape[1]-offset, 0]
        ])

    #perspective transformed
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

#PERCENTILE
def threshold_precentile(immagine, percentile=98):
    high = np.percentile(immagine, percentile)
    threshold = int(high)
    mask = cv2.inRange(immagine, (threshold), (255))
    return mask

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if (orient == 'x'):
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_soble = np.uint8(255*sobel/np.max(sobel))
    sxbinary = np.zeros_like(scaled_soble)
    sxbinary[(scaled_soble >= thresh[0]) & (scaled_soble <= thresh[1])] = 1
    binary_output = sxbinary
    return binary_output

def mag_thresh(img, sobel_kernel, mag_thresh=(0, 255)):
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_magnitude = np.uint8(255*magnitude/np.max(magnitude))
    # 5) Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_magnitude)
    sxbinary[(scaled_magnitude >= mag_thresh[0]) & (scaled_magnitude <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = sxbinary # Remove this line
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    sxbinary = np.zeros_like(direction)
    sxbinary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = sxbinary # Remove this line
    return binary_output

def perform_threshold_mask(immagine_per_mask, percentile=98):
    # create images in different color spaces
    hsv = cv2.cvtColor(immagine_per_mask, cv2.COLOR_RGB2HSV)
    hls = cv2.cvtColor(immagine_per_mask, cv2.COLOR_RGB2HLS)
    luv = cv2.cvtColor(immagine_per_mask, cv2.COLOR_RGB2LUV)
    lab = cv2.cvtColor(immagine_per_mask, cv2.COLOR_RGB2LAB)
    xyz = cv2.cvtColor(immagine_per_mask, cv2.COLOR_RGB2XYZ)
    yuv = cv2.cvtColor(immagine_per_mask, cv2.COLOR_RGB2YUV)

    #use only the one we think is good (done in notebook)
    #RGB - RG
    final_mask= threshold_precentile(immagine_per_mask[:,:,0])
    final_mask=final_mask+threshold_precentile(immagine_per_mask[:,:,1])
    #HSV - V
    final_mask=final_mask+threshold_precentile(hsv[:,:,2])
    #LUV - UV
    final_mask=final_mask+threshold_precentile(luv[:,:,1])
    final_mask=final_mask+threshold_precentile(luv[:,:,2])
    #LAB - LB
    final_mask=final_mask+threshold_precentile(lab[:,:,0])
    final_mask=final_mask+threshold_precentile(lab[:,:,2])
    # XY - ALL
    final_mask=final_mask+threshold_precentile(xyz[:,:,0])
    final_mask=final_mask+threshold_precentile(xyz[:,:,1])
    # YUV - U
    final_mask=final_mask+threshold_precentile(yuv[:,:,1])
    return final_mask

def perform_sobel_mask(immagine_per_sobel, ksize=5):
    hls = cv2.cvtColor(immagine_per_sobel, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    yuv = cv2.cvtColor(immagine_per_sobel, cv2.COLOR_RGB2YUV)
    u_channel = 255 - yuv[:,:,1]
    v_channel = 255 - yuv[:,:,2]

    #gray = cv2.cvtColor(immagine_per_sobel, cv2.COLOR_BGR2GRAY)

    combined_all = np.stack((u_channel, v_channel, s_channel), axis=2)
    uvs = np.uint8(np.mean(combined_all, 2))

    #operations
    gradx_gray = abs_sobel_thresh(uvs, orient='x', sobel_kernel=3, thresh=(25, 255))
    grady_gray  = abs_sobel_thresh(uvs, orient='y', sobel_kernel=3, thresh=(25, 255))
    mag_binary_gray = mag_thresh(uvs, sobel_kernel=ksize, mag_thresh=(30, 512))
    dir_binary_gray = dir_threshold(uvs, sobel_kernel=3, thresh=(0.2, 1 ))

    combined_gray = np.zeros_like(dir_binary_gray)
    combined_gray[((gradx_gray == 1)&(grady_gray == 1)) | ((mag_binary_gray == 1) & (dir_binary_gray == 1))] = 1
    return np.uint8(combined_gray*255)

def find_curvature_init(binary_warped):
    # Prerequisite: we have created a warped binary image called "binary_warped"
    # Parameters:
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- marginx
    marginx = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Take a histogram of the bottom half of the image
    half_image=int(binary_warped.shape[0]/2)

    histogram = np.sum(binary_warped[half_image:,:], axis=0)
    #histogram = np.sum(binary_warped, axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype(np.uint8)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - marginx
        win_xleft_high = leftx_current + marginx
        win_xright_low = rightx_current - marginx
        win_xright_high = rightx_current + marginx
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Determine the lane curvature
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #VISUALIZATION OF THE RESULT
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return left_fit, right_fit, ploty, left_fitx, right_fitx, out_img

def find_curvature(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype(np.uint8)
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return left_fit, right_fit, ploty, left_fitx, right_fitx, result

def curvature_radius(ploty, leftx, rightx):
    y_eval = 720
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/753 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad
    # Example values: 632.1 m    626.2 m

def project_back(warped, ploty, leftx, rightx, undistorted_image):
    offset=350
    # FOUR SOURCE COORDINATES
    src = np.float32(
        [[328, 670],
        [1081, 670],
        [601, 450],
        [687, 450]])

    # FOUR DESTINATION COORDINATES
    dst = np.float32(
        [
        [src[0][0], img.shape[0]],
        [src[1][0], img.shape[0]],
        [src[0][0], 0],
        [src[1][0], 0]
        ])

    # better
    dst = np.float32(
        [
        [offset, img.shape[0]],
        [img.shape[1]-offset, img.shape[0]],
        [offset, 0],
        [img.shape[1]-offset, 0]
        ])

    Minv = cv2.getPerspectiveTransform(dst, src)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_image.shape[1], undistorted_image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)
    return result


# SINITIALIZATION

# CALIBRATION: to perform only if not alreadu done
camera_cal_file = Path("output_images/camera_cal_pickle.p")
condition = camera_cal_file.is_file()
# uncomment next line to force calibration
#condition = False
if condition:
    print ("Camera calibration already done")
else:
    # PERFORM AND TEST CAMERA CALIBRATION
    camera_calibration()
    verify_camera_calibration()

# PERSPECTIVE TRANSFORM INITIALIZAZION
# extract an image from video for setting points
# only once, when calibration has just been done
# for using to define the perspective transform
#condition = False
if not condition:
    print ("Extracting an image of straight road...")
    straight_road_img = extract_frame(15,'project_video.mp4')
    cv2.imwrite('output_images/frame347.jpg',straight_road_img)
    straight_road_img_undistorted = distortion_correction("output_images/camera_cal_pickle.p", straight_road_img)
    cv2.imwrite('output_images/frame347_undist.jpg',straight_road_img_undistorted)
    straight_road_img_undistorted_warped=warp_frame(straight_road_img_undistorted)
    cv2.imwrite('./output_images/frame347_undist_warped.jpg',straight_road_img_undistorted_warped)
    print ("Image saved")


img = mpimg.imread("output_images/frame347_undist_warped.jpg")
binary_warped = perform_threshold_mask(img)

figure1 = plt.figure(1)
plt.subplot(221)
plt.imshow(img)
plt.title('Original warped')

plt.subplot(222)
plt.imshow(binary_warped, cmap='gray')
plt.title('Binary')


left_fit, right_fit, ploty, left_fitx, right_fitx, out_img = find_curvature_init(binary_warped)
left_cv, right_cv=curvature_radius(ploty, left_fitx, right_fitx)
plt.subplot(223)
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.title('Lines')

plt.subplot(224)
straight_road_img_undistorted = mpimg.imread("output_images/frame347_undist.jpg")
image_back=project_back(binary_warped, ploty, left_fitx, right_fitx, straight_road_img_undistorted)
left_cv, right_cv=curvature_radius(ploty, left_fitx, right_fitx)
deviation_from_center = "Deviation from center (right if >0): {:.2f} meters".format(np.mean((right_fitx[-50:] + left_fitx[-50:])/2-binary_warped.shape[1]/2)*3.7/753)
average_curvature = "Curvature: {:.1f} meters".format((left_cv + right_cv)/2)
txt = deviation_from_center + "\n" + average_curvature
plt.title('Lanes, curvature, and position')

plt.imshow(image_back)
figure1.text(0.35, 0.48, txt, style='italic',fontsize=8, bbox={'facecolor':'red', 'alpha':0.5, 'pad':8})
plt.savefig('./output_images/treatment_on_single_frame.png')
plt.show()

print ("Saving image....treatment_on_single_frame.png")
#cv2.imwrite('./output_images/treatment_on_single_frame.jpg',dir_binary)


# MAIN --------

#video = 'harder_challenge_video.mp4'
#video = mpy.VideoFileClip("project_video.mp4").subclip(38,43)
video = 'project_video.mp4'
cap = cv2.VideoCapture(video)
ret, frame = cap.read()

current_frame=0
counter_bad_frame=0
stacked_left_fit=[]
stacked_right_fit=[]
stacked_ploty=[]
stacked_left_fitx=[]
stacked_right_fitx=[]
stacked_left_cv=[]
stacked_right_cv=[]

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #macosx

video_out = cv2.VideoWriter('franz_output.mp4',fourcc, 20.0,(frame.shape[1],frame.shape[0]))

print ("Working on video and saving it")
while(cap.isOpened()):
    ret, frame = cap.read()

    undistorted_frame = distortion_correction("output_images/camera_cal_pickle.p",frame)
    warped_frame = warp_frame(undistorted_frame)
    binary_frame = perform_threshold_mask(cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB), 98)
    #binary_frame_old = perform_sobel_mask(cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB), 5)

    if current_frame==0:
        condition = True
    # check wich kind of detection we have to use
    if condition:
        left_fit, right_fit, ploty, left_fitx, right_fitx, out_img2=find_curvature_init(binary_frame)
    else:
        left_fit, right_fit, ploty, left_fitx, right_fitx, out_img2=find_curvature(binary_frame, left_fit, right_fit)

    #compute curvature radius in real world
    left_cv, right_cv=curvature_radius(ploty, left_fitx, right_fitx)

    # SANITY CHECK

    delta_curvature = 0.5 #50%
    delta_distance = 0.05 #5%
    delta_parallel = 50 #pixels
    if (
        #curvature
        (left_cv > (right_cv+right_cv*delta_curvature))| (left_cv < (right_cv-right_cv*delta_curvature))
        #distance
        |(  (np.mean(right_fitx[-50:] - left_fitx[-50:])  < np.mean(570*(1-delta_distance))) | (np.mean(right_fitx[-50:] - left_fitx[-50:]) > np.mean(570*(1+delta_distance))))
        #parallel
        |( abs(np.mean(right_fitx[-50:] - left_fitx[-50:]) - np.mean(right_fitx[:50] - left_fitx[:50]))>100)
        ):
        print("bad data ", current_frame)
        counter_bad_frame=counter_bad_frame+1
        if counter_bad_frame > 1:
            condition = True
    else:

        counter_bad_frame=0
        condition = False

    #ADD VALUES TO STACK
    stacked_left_fit.append(left_fit)
    stacked_right_fit.append(right_fit)
    stacked_ploty.append(ploty)
    stacked_left_fitx.append(left_fitx)
    stacked_right_fitx.append(right_fitx)
    stacked_left_cv.append(left_cv)
    stacked_right_cv.append(right_cv)

    #smoothing measurements

    last_frames=len(stacked_left_fitx)
    if last_frames >1:
        last_frames = 1
    computed_ploty = np.mean(stacked_ploty[-last_frames:],0)
    computed_left_fitx = np.mean(stacked_left_fitx[-last_frames:],0)
    computed_right_fitx = np.mean(stacked_right_fitx[-last_frames:],0)
    computed_left_cv = np.mean(stacked_left_cv[-last_frames:],0)
    computed_right_cv = np.mean(stacked_right_cv[-last_frames:],0)
    # >0 too much right
    deviation_from_center = "Deviation from center (right if >0): {:.2f} meters".format(np.mean((computed_right_fitx[-50:] + computed_left_fitx[-50:])/2-binary_warped.shape[1]/2)*3.7/753)
    average_curvature = "Curvature: {:.1f} meters".format((computed_left_cv + computed_right_cv)/2)


    #Drawing the lines back down onto the road
    last = project_back(binary_frame, computed_ploty, computed_left_fitx, computed_right_fitx, undistorted_frame)
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(last,average_curvature,(10,50), font, 1,(255,255,255),2)
    cv2.putText(last,deviation_from_center,(10,80), font, 1,(255,255,255),2)



    # Convert grayscale image to 3-channel image,so that they can be stacked together
    #imgc_binary_frame = cv2.cvtColor(binary_frame,cv2.COLOR_GRAY2BGR)
    #imgc_binary_frame_old = cv2.cvtColor(binary_frame_old,cv2.COLOR_GRAY2BGR)
    #both = np.hstack((imgc_binary_frame,last))


    #displayed_img=cv2.resize(both,  None, fx=0.5, fy=0.5);
    cv2.imshow('WindowName',last)
    #video_out.write(last)
    current_frame = current_frame +1
    if (current_frame%50) == 0:
        print ("Siamo al ", current_frame/(5*2.5), "%")




    if (cv2.waitKey(25) & 0xFF == ord('q')):
        break


cap.release()
video_out.release()
cv2.destroyAllWindows()
