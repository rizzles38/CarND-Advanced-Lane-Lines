import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt

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


#undistorting test images
test_images = glob.glob("./test_images/test*.jpg")

for idx, fname in enumerate(test_images):
    img = cv2.imread(fname)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    write_name = "./test_images/tracked"+str(idx)+".jpg"
    cv2.imwrite(write_name, img)

#convert to HLS
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    (hue, lit, sat) = cv2.split(img)

# use sobel on both x and y
    sobelx = cv2.Sobel(sat, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(sat, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))

    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= thresh_min) & (scaled_sobely <= thresh_max)] = 1

    both_binary = np.zeros_like(sxbinary)
    both_binary[(sxbinary == 1) & (sybinary ==1)] = 1



    #plt.imshow(sxbinary, cmap = "gray")
    #plt.show()
    plt.imshow(both_binary, cmap = "gray")
    plt.show()
    exit()

