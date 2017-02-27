import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, ImageSequenceClip

def search_prev_pos(binary_warped, avg_left_fit, avg_right_fit, med_left_fit, med_right_fit):
    if (len(avg_left_fit) == 0):
        print("left_fit empty")
        return None, None, None, None
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (avg_left_fit[0]*(nonzeroy**2) + avg_left_fit[1]*nonzeroy + avg_left_fit[2] - margin)) & (nonzerox < (avg_left_fit[0]*(nonzeroy**2) + avg_left_fit[1]*nonzeroy + avg_left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (avg_right_fit[0]*(nonzeroy**2) + avg_right_fit[1]*nonzeroy + avg_right_fit[2] - margin)) & (nonzerox < (avg_right_fit[0]*(nonzeroy**2) + avg_right_fit[1]*nonzeroy + avg_right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    print("len(leftx): ", len(leftx)) #for testing purposes only
    print("len(rightx): ", len(rightx)) #for texitng purposes only

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    #check if our left and right fit coefficients are too far off
    #if too far off, consider them outliers and don't include them - instead use median fit that was passed in
    outlier_detected = False
    #print("new right fit: {0:.6f} {1:.6f} {2:.6f}".format(right_fit[0], right_fit[1], right_fit[2]))
    if med_right_fit is not None:
        #print("median right fit: {0:.6f} {1:.6f} {2:.6f}".format(med_right_fit[0], med_right_fit[1], med_right_fit[2]))
        # if we have an outlier:
        if (abs(right_fit[0]) > 2 * abs(med_right_fit[0])) or (abs(right_fit[0]) < abs(med_right_fit[0])/2):  #outlier
            outlier_detected = True
        if (abs(left_fit[0]) > 2 * abs(med_left_fit[0])) or (abs(left_fit[0]) < abs(med_left_fit[0])/2):
            outlier_detected = True # TODO have separate flags for each lane
        if (right_fit[2] < left_fit[2]):
            outlier_detected = True
        # still return actually left_fit as left_fit
            # use median to get left_fitx
    # if med_left_fit is not empty, check for outliers.  if empty, do nothing and use the left fit we just obtained
    
    # visualize the result
    # Generate x and y values for plotting
    if not outlier_detected:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    else:
        left_fitx = med_left_fit[0]*ploty**2 + med_left_fit[1]*ploty + med_left_fit[2]
        right_fitx = med_right_fit[0]*ploty**2 + med_right_fit[1]*ploty + med_right_fit[2]
    left_lane = np.stack((left_fitx, ploty), axis=-1)
    print(left_lane)
    right_lane = np.stack((right_fitx, ploty), axis=-1)
    return left_lane, right_lane, left_fit, right_fit
 


    #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #left_lane = np.stack((left_fitx, ploty), axis=-1)
    #right_lane = np.stack((right_fitx, ploty), axis=-1)
    #return left_lane, right_lane, left_fit, right_fit
 

def sliding_window(binary_warped, avg_left_fit, avg_right_fit, med_left_fit, med_right_fit):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    startpoint = np.int(0.20 * histogram.shape[0])
    midpoint = np.int(histogram.shape[0]/2)
    offset = np.int(0.10 * histogram.shape[0])
    leftx_base = np.argmax(histogram[startpoint:midpoint-offset]) + startpoint
    rightx_base = np.argmax(histogram[midpoint+offset:len(histogram)-startpoint]) + midpoint + offset

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
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

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except TypeError:
        left_fit = (0, 0, leftx_base)
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except TypeError:
        right_fit = (0, 0, rightx_base)


    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    #check if our left and right fit coefficients are too far off
    #if too far off, consider them outliers and don't include them - instead use median fit that was passed in
    outlier_detected = False
    #print("new right fit: {0:.6f} {1:.6f} {2:.6f}".format(right_fit[0], right_fit[1], right_fit[2]))
    if med_right_fit is not None:
        #print("median right fit: {0:.6f} {1:.6f} {2:.6f}".format(med_right_fit[0], med_right_fit[1], med_right_fit[2]))
        # if we have an outlier:
        if (abs(right_fit[0]) > 2 * abs(med_right_fit[0])) or (abs(right_fit[0]) < abs(med_right_fit[0])/2):  #outlier
            outlier_detected = True
        if (abs(left_fit[0]) > 2 * abs(med_left_fit[0])) or (abs(left_fit[0]) < abs(med_left_fit[0])/2):
            outlier_detected = True # TODO have separate flags for each lane
        if (right_fit[2] < left_fit[2]):
            outlier_detected = True
        # still return actually left_fit as left_fit
            # use median to get left_fitx
    # if med_left_fit is not empty, check for outliers.  if empty, do nothing and use the left fit we just obtained
    
    # visualize the result
    # Generate x and y values for plotting
    if not outlier_detected:
        # average the fits with previous frames
        # there MUST be a better way to do this
        if (len(avg_left_fit) > 0):
            first_left_coeff = (left_fit[0] + avg_left_fit[0])/2
            second_left_coeff = (left_fit[1] + avg_left_fit[1])/2
            third_left_coeff = (left_fit[2] + avg_left_fit[2])/2 
            smoothed_left_fit = [first_left_coeff, second_left_coeff, third_left_coeff]
            first_right_coeff = (right_fit[0] + avg_right_fit[0])/2
            second_right_coeff = (right_fit[1] + avg_right_fit[1])/2
            third_right_coeff = (right_fit[2] + avg_right_fit[2])/2
            smoothed_right_fit = [first_right_coeff, second_right_coeff, third_right_coeff]
        else:  #if we don't have anything to average with yet, use the one we just found
            smoothed_left_fit = left_fit
            smoothed_right_fit = right_fit
        left_fitx = smoothed_left_fit[0]*ploty**2 + smoothed_left_fit[1]*ploty + smoothed_left_fit[2]
        right_fitx = smoothed_right_fit[0]*ploty**2 + smoothed_right_fit[1]*ploty + smoothed_right_fit[2]
    else:
        left_fitx = med_left_fit[0]*ploty**2 + med_left_fit[1]*ploty + med_left_fit[2]
        right_fitx = med_right_fit[0]*ploty**2 + med_right_fit[1]*ploty + med_right_fit[2]
    left_lane = np.stack((left_fitx, ploty), axis=-1)
    print(left_lane)
    right_lane = np.stack((right_fitx, ploty), axis=-1)
    return left_lane, right_lane, left_fit, right_fit

def draw_road(img, Minv, left_lane, right_lane):
    overlay = img.copy()

    warped_left = cv2.perspectiveTransform(np.array([left_lane]), Minv)[0].astype(int)
    warped_right = cv2.perspectiveTransform(np.array([right_lane]), Minv)[0].astype(int)

    poly = np.array([np.concatenate((warped_left, warped_right[::-1]), axis=0)])

    # draw polygon between lane lines
    cv2.fillPoly(overlay, poly, (0, 255, 0, 100))

    # blend overlay and original image
    alpha = 0.34
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    # draw left lane
    for i in range(1, len(warped_left)):
        cv2.line(img,
                 (warped_left[i - 1][0], warped_left[i - 1][1]),
                 (warped_left[i][0], warped_left[i][1]),
                 (255, 0, 0), 3)

    # draw right lane
    for i in range(1, len(warped_right)):
        cv2.line(img,
                 (warped_right[i - 1][0], warped_right[i - 1][1]),
                 (warped_right[i][0], warped_right[i][1]),
                 (0, 0, 255), 3)

# calibrate camera
object_pt = np.zeros((6*9, 3), np.float32)
object_pt[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

images = glob.glob("./camera_cal/calibration*.jpg")

print(images[0])

objpoints = []
imgpoints = []

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray)
    # plt.show()
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret is True:
        objpoints.append(object_pt)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        write_name = "drawn_corners"+str(idx)+".jpg"
        cv2.imwrite("./camera_cal/"+write_name, img)

        # No idea if I'm doing this right.  Should I calibrate the camera fro every image in a loop?
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#plt.ion()
#plt.show()

in_clip = VideoFileClip("project_video.mp4")
out_frames = []
frame_count = 0
frame_start = 0
buf_size = 7
right_fit = []
left_fit = [] 
avg_left_fit = []
avg_right_fit = []
last_n_left_fits = []
last_n_right_fits = []
for in_frame in in_clip.iter_frames():
    if frame_count < frame_start:  #for testing purposes only
        frame_count += 1
        continue
    img = cv2.undistort(in_frame, mtx, dist, None, mtx)
    orig_img = img.copy()

    img_height = img.shape[0]
    img_width = img.shape[1]

    # define values to scale by
    bot_width_pct = .76
    mid_width_pct = .08
    height_pct = .62
    bottom_trim_pct = .935

    #verify the signs here:                                                 
    center_x = img_width * 0.5
    top_y = img_height * height_pct
    top_x = mid_width_pct * img_width / 2
    bot_x = bot_width_pct * img_width / 2
    #do the transform
    src = np.float32([[center_x-top_x, top_y],[center_x+top_x, top_y],[center_x+bot_x,bottom_trim_pct*img_height],[center_x-bot_x,bottom_trim_pct*img_height]]) #fill with points
    offset = 0.25 * img_width
    dst = np.float32([[offset, 0], [img_width-offset, 0], [img_width-offset, img_height], [offset, img_height]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #img = cv2.warpPerspective(img, M, (img_width, img_height), flags=cv2.INTER_LINEAR)
    img = cv2.warpPerspective(img, M, (img_width, img_height), flags=cv2.INTER_NEAREST)

    #plt.imshow(img)
    #plt.draw()
    #plt.pause(0.001)

    # convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    (hue, lit, sat) = cv2.split(hls)
    # convert the sat image to binary
    sat_thresh_min = 150
    sat_thresh_max = 255
    sat_binary = np.zeros_like(sat)
    sat_binary[(sat >= sat_thresh_min) & (sat <= sat_thresh_max)] = 1

    hue_thresh_min = 18
    hue_thresh_max = 24
    hue_binary = np.zeros_like(hue)
    hue_binary[(hue >= hue_thresh_min) & (hue <= hue_thresh_max)] = 1

    hls_binary = np.zeros_like(sat)
    #hls_binary[(sat_binary == 1) & (hue_binary == 1)] = 1
    hls_binary[(sat_binary == 1)] = 1

    # use sobel on both x and y
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    thresh_min = 30
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1

    # fill in these
    combined_binary = np.zeros_like(hls_binary)
    combined_binary[(hls_binary == 1) | (sxbinary == 1)] = 1

   #if () # we didn't find lines in previous frames
    #else:

    #get the medians of left fits and right fits
    if last_n_left_fits:   #if I don't have any fits yet, nothing to sort, set to None
        left_fits_copy = sorted(last_n_left_fits, key=lambda coeffs: coeffs[0])
        med_left_fit = left_fits_copy[len(left_fits_copy)//2]
    else:
        med_left_fit = None
    
    if last_n_right_fits:
        right_fits_copy = sorted(last_n_right_fits, key=lambda coeffs: coeffs[0])
    #find the median
        med_right_fit = right_fits_copy[len(right_fits_copy)//2]
    else:
        med_right_fit = None
     #instead of sliding window, just to search based on last 5 frames
    #if len(last_n_left_fits) > 0:
        # calculate average left and right fits over the last n frames
    first_left_coeff_avg =  sum(coefficients[0] for coefficients in last_n_left_fits)/buf_size
    second_left_coeff_avg = sum(coefficients[1] for coefficients in last_n_left_fits)/buf_size 
    third_left_coeff_avg = sum(coefficients[2] for coefficients in last_n_left_fits)/buf_size 
    avg_left_fit = [first_left_coeff_avg, second_left_coeff_avg, third_left_coeff_avg]
    first_right_coeff_avg = sum(coeffs[0] for coeffs in last_n_right_fits)/buf_size
    second_right_coeff_avg = sum(coeffs[1] for coeffs in last_n_right_fits)/buf_size
    third_right_coeff_avg = sum(coeffs[2] for coeffs in last_n_right_fits)/buf_size
    avg_right_fit = [first_right_coeff_avg, second_right_coeff_avg, third_right_coeff_avg]

        #try without the other function first
        #left_lane, right_lane, left_fit, right_fit = search_prev_pos(combined_binary, avg_left_fit, avg_right_fit, med_left_fit, med_right_fit)
    #else:  # use sliding window for first frame 
    left_lane, right_lane, left_fit, right_fit = sliding_window(combined_binary, avg_left_fit, avg_right_fit, med_left_fit, med_right_fit)
    if frame_count == frame_start:
        last_n_left_fits = [left_fit] * buf_size
        last_n_right_fits = [right_fit] * buf_size
    else:
        last_n_left_fits.append(left_fit)
        last_n_left_fits.pop(0)
        last_n_right_fits.append(right_fit)
        last_n_right_fits.pop(0)
    # draw the road
    draw_road(orig_img, Minv, left_lane, right_lane)

    #plt.imshow(orig_img) #for testing only
    #plt.show()                            #for testing only
    out_frames.append(orig_img)
    print("Processed frame {}".format(frame_count))
    frame_count += 1
    
    if frame_count == 1200: # max number of frames to process
        break

out_clip = ImageSequenceClip(out_frames, fps=in_clip.fps)
out_clip.write_videofile("output.mp4")
