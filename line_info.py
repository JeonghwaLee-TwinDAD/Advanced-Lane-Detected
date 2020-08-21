import glob
import pickle
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def cal_camera(images, nx=6, ny=9):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:ny,0:nx].T.reshape(-1,2)
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (ny,nx),None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (ny,nx), corners, ret)

    return objpoints, imgpoints

def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist, mtx, dist    

def warp(img):
    height, width = img.shape[:2]
            
    xshift = 150 # x shift from 0 at x-axis 
    yshift = 100 # y shift from image center 
    xoffset = 0 # trapzoidal box shift to x-direction
    yoffset = 10 # trapzoidal box shift to y-direction

    src = np.float32([[(width - xshift)/2, height/2 + yshift],
                     [(width + xshift)/2, height/2 + yshift],
                     [width - xshift, height],
                     [xshift, height]])

    dst = np.float32([[width/4, yoffset],
                      [width*3/4, yoffset],
                      [width*3/4, height],
                      [width/4, height]])
    
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse   
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)
    
    return warped, Minv

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary_output

# morphological transformations to highlight lines of interest even more and remove noise.
def color_threthold(img):
    ## Choose Color theshold method (white) 
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    white = np.zeros_like(l_channel)
    white[(l_channel > 200) & (l_channel <= 255)] = 1    

     # Lab B-channel Threshold (using default parameters)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b_channel = lab[:,:,2]
    yellow = np.zeros_like(b_channel)
    yellow[(b_channel > 150) & (b_channel <= 255)] = 1    

    color_binary = np.zeros_like(yellow)
    color_binary[(yellow == 1) | (white == 1)] = 1

    # remove noise
    color_binary = cv2.morphologyEx(color_binary, cv2.MORPH_OPEN, kernel=np.ones((5,5),dtype=np.uint8))
    # close mask
    color_binary = cv2.morphologyEx(color_binary, cv2.MORPH_CLOSE, kernel=np.ones((5,5),dtype=np.uint8))
   
    return color_binary

def window_search(binary_warped):
    img_size = binary_warped.shape
    # Take a histogram of the bottom 2/3rd of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]*2//3:,:], axis=0)    
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
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
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2)
       
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.average(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.average(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    try:
        ploty = np.linspace(0, img_size[0]-1, img_size[0])
        # Fit new polynomials
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fit = [0,0,0]
        right_fit = [0,0,0]
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        #left_fit = 1*ploty**2 + 1*ploty
        #right_fit = 1*ploty**2 + 1*ploty
            
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    '''
    # Plots histogram and the left & right polynomials on the lane lines
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    f.set_figheight(5)
    ax1.plot(histogram)
    ax1.set_title('Histogram', fontsize=30)
    ploty = np.linspace(0, img_size[0]-1, img_size[0])
    ax2.plot(left_fitx, ploty, color='yellow')
    ax2.plot(right_fitx, ploty, color='yellow')
    ax2.imshow(out_img)
    ax2.set_title('window sliding', fontsize=30)
    '''
    return out_img, left_fit, right_fit, leftx, lefty, rightx, righty  

def poly_search(binary_warped, left_fit_prev, right_fit_prev):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    img_size = binary_warped.shape
    left_fit = left_fit_prev
    right_fit = right_fit_prev
    margin = 80 # defined searching band (boundary margin)

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    try:
        ploty = np.linspace(0, img_size[0]-1, img_size[0])
        # Fit new polynomials
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fit = [0,0,0]
        right_fit = [0,0,0]
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        #left_fit = 1*ploty**2 + 1*ploty
        #right_fit = 1*ploty**2 + 1*ploty
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255] 
  
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    poly_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    '''         
    # Plots the left and right polynomials on the lane lines
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)

    ax1.imshow(binary_warped, cmap='gray')
    ax1.set_title('Birds-eye view', fontsize=30)
    ax2.plot(left_fitx, ploty, color='yellow')
    ax2.plot(right_fitx, ploty, color='yellow')
    ax2.imshow(poly_img)
    ax2.set_title('polynominal search tracking', fontsize=30)
    '''
    return poly_img, left_fit, right_fit, leftx, lefty, rightx, righty

# Method to determine radius of curvature and distance from lane center 
# based on binary image, polynomial fit, and L and R lane pixel indices
def curv_rad(img_size, x, y, xm_per_pix, ym_per_pix = 3.048/100):
    ploty = np.linspace(0, img_size[0]-1, img_size[0])
    y_eval = np.max(ploty)
    
    if len(x) != 0:
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        # Now our radius of curvature is in meters
        
    return curverad

def draw_lane(original_img, binary_img, left_fit, right_fit, Minv):
    new_img = np.copy(original_img)
    
    if left_fit is None or right_fit is None:
        return original_img
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h,w = (binary_img.shape[0], binary_img.shape[1])
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image.
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    polygon = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    
    return polygon

def draw_data(original_img, curv_rad, center_dist):
    new_img = np.copy(original_img)
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return new_img

def pipeline(img):
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load( open( "distortion_pickle.p", "rb" ) )

    objpoints = dist_pickle["objpoints"]
    imgpoints = dist_pickle["imgpoints"]
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    undistorted = cv2.undistort(img, mtx, dist, None, mtx)# step #1 : undistorted
    warped, Minv = warp(undistorted) # step #3 : Warped binary image  
    binary = color_threthold(warped) # step #2 : gradient, color transform, and noise filtering
    
    return binary, Minv

def plot_fit_onto_img(img, fit, plot_color):
    if fit is None:
        return img
    new_img = np.copy(img)
    h = new_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    plotx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
    cv2.polylines(new_img, np.int32([pts]), isClosed=False, color=plot_color, thickness=8)
    return new_img

def region_of_interest(img, warped_img):
    xshift = 150 # x shift from 0 at x-axis 
    yshift = 100 # y shift from image center 
    xoffset = 0 # trapzoidal box shift to x-direction
    yoffset = 10 # trapzoidal box shift to y-direction

    src = np.float32([[(img.shape[1] - xshift)/2, img.shape[0]/2 + yshift],
                     [(img.shape[1] + xshift)/2, img.shape[0]/2 + yshift],
                     [img.shape[1] - xshift, img.shape[0]],
                     [xshift, img.shape[0]]])

    dst = np.float32([[img.shape[1]/4, 10],
                      [img.shape[1]*3/4, 10],
                      [img.shape[1]*3/4, img.shape[0]],
                      [img.shape[1]/4, img.shape[0]]])    
    
    # draw the polynominal in the original image
    cv2.polylines(img, np.int32([src]), isClosed=True, color=(255,0,0), thickness=5)
    cv2.polylines(warped_img, np.int32([dst]), isClosed=True, color=(255,0,0), thickness=5)

    # Visualize unwarp
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Region of Interest', fontsize=30)
    ax2.imshow(warped_img)
    ax2.set_title('Unwarped', fontsize=30)

def diagonstic(draw_img, bin_img, l_line , r_line, Minv):
    color_ok = (200,255,155)
    color_bad = (255,155,155)
    font = cv2.FONT_HERSHEY_DUPLEX

    # put together multi-view output
    diag_screen = np.zeros((720,1280,3), dtype=np.uint8)

    # original output display(top left)
    diag_screen[0:360,0:640,:] = cv2.resize(draw_img,(640,360))
   
    # binary overhead view (bottom right)
    unwarp_bin = np.dstack((bin_img, bin_img, bin_img))*255
    newwarp = cv2.warpPerspective(unwarp_bin, Minv, (bin_img.shape[1], bin_img.shape[0])) 
    diag_screen[360:720,640:1280, :] = cv2.resize(newwarp,(640,360)) 
  
    # binary overhead view (bottom right)
    bin_img = np.copy(bin_img)
    fit_bin = np.dstack((bin_img, bin_img, bin_img))*255

    if l_line.current_fit is not None:
        for i, fit in enumerate(l_line.current_fit):
            fit_bin = plot_fit_onto_img(fit_bin, fit, (20*i+100,0,20*i+100))
        fit_bin = plot_fit_onto_img(fit_bin, l_line.best_fit, (255,255,0))
    else:
        cv2.putText(diag_screen, "No left fit in the buffer", (660,400), font, .7, color_bad, 1, cv2.LINE_AA)
    
    if r_line.current_fit is not None:
        for i, fit in enumerate(r_line.current_fit):
            fit_bin = plot_fit_onto_img(fit_bin, fit, (0,20*i+100,20*i+100))
        fit_bin = plot_fit_onto_img(fit_bin, r_line.best_fit, (255,255,0))
    else:
        cv2.putText(diag_screen, "No right fit in the buffer", (1000,400), font, .7, color_bad, 1, cv2.LINE_AA)

    diag_screen[0:360,640:1280, :] = cv2.resize(fit_bin,(640,360))

    ## Diagnostic data (bottom left)
    # Draw data of best fit 
    if l_line.best_fit is not None:
        text = 'Best fit L: ' + ' {:0.6f}'.format(l_line.best_fit[0]) + \
                                ' {:0.6f}'.format(l_line.best_fit[1]) + \
                                ' {:0.6f}'.format(l_line.best_fit[2])
        color_info = color_ok
    else:
        text = "No best fit Left lane"
        color_info = color_bad
    cv2.putText(diag_screen, text, (20,400), font, .7, color_info, 1, cv2.LINE_AA)

    if r_line.best_fit is not None:
        text = 'Best fit R: ' + ' {:0.6f}'.format(r_line.best_fit[0]) + \
                                ' {:0.6f}'.format(r_line.best_fit[1]) + \
                                ' {:0.6f}'.format(r_line.best_fit[2])
        color_info = color_ok
    else:
        test = "No best fit right lane"
        color_info = color_bad
    cv2.putText(diag_screen, text, (20,420), font, .7, color_info, 1, cv2.LINE_AA)

    # Draw Best x postiopn 
    if l_line.line_base_pos is not None:
        text = 'Left lane position(m): ' + str('{:0.6f}'.format(l_line.line_base_pos))
        color_info = color_ok
    else:
        text = 'There is no left lane'
        color_info = color_bad
    cv2.putText(diag_screen, text, (20,460), font, .7, color_info, 1, cv2.LINE_AA)

    if r_line.line_base_pos is not None:
        text = 'Right lane position(m):  ' + str('{:0.6f}'.format(r_line.line_base_pos))
        color_info = color_ok
    else:
        text = 'There is no right lane'
        color_info = color_bad
    cv2.putText(diag_screen, text, (20,480), font, .7, color_info, 1, cv2.LINE_AA)

    # Draw number of current fit list (n <= 20)
    if l_line.current_fit is not None:
        text = 'Good fit count L: ' + str(len(l_line.current_fit))
        color_info = color_ok
    else:
        text = 'Empty in the left current fit list'
        color_info = color_bad
    cv2.putText(diag_screen, text, (20,520), font, .7, color_info, 1, cv2.LINE_AA)

    if r_line.current_fit is not None:
        text = 'Good fit count R: ' + str(len(r_line.current_fit))
        color_info = color_ok
    else:
        text = 'Empty in the right current fit list'
        color_info = color_bad
    cv2.putText(diag_screen, text, (20,540), font, .7, color_info, 1, cv2.LINE_AA)
    
    # Lane base constraint (meter) 
    lane_base = (r_line.line_base_pos - l_line.line_base_pos)
    if lane_base > 3 and lane_base < 4:
        text = 'Lane base (3m < x < 4 m ): ' + str('{:0.6f}'.format(lane_base))
        color_info = color_ok
    else:
        text = 'Lane base is out of range'
        color_info = color_bad
    cv2.putText(diag_screen, text, (20,580), font, .7, color_info, 1, cv2.LINE_AA)
    
    # Redius of curvature constraint 
    if l_line.radius_of_curvature < 10000 and r_line.radius_of_curvature < 10000:
        del_curv = abs(l_line.radius_of_curvature - r_line.radius_of_curvature)
        text = 'Delta Radius of Curvature < 500 :  ' + str(' {:0.6f}'.format(del_curv))
        cv2.putText(diag_screen, text, (20,620), font, .7, color_ok, 1, cv2.LINE_AA)
    else:
        test = "Driving through strait ahead (r_curv > 10,000 m)"                
        cv2.putText(diag_screen, text, (20,640), font, .7, color_bad, 1, cv2.LINE_AA)
    
    return diag_screen

'''
# Color and gradient threshold testing
#img= cv2.imread('./challenge_images/challenge_01.jpg')
img= cv2.imread('./test_images/test1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

dist_pickle = pickle.load( open( "distortion_pickle.p", "rb" ) )

objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

undistorted = cv2.undistort(img, mtx, dist, None, mtx)# step #1 : undistorted
warped, Minv = warp(undistorted) # step #3 : Warped binary image  
#region_of_interest(img, undistorted)
binary = color_threthold(warped) # step #2 : gradient, color transform, and noise filtering


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
f.tight_layout()
ax1.imshow(warped)
ax1.axis('off')
ax1.set_title('original', fontsize=20)
ax2.imshow(binary, cmap='gray')
ax2.axis('off')
ax2.set_title('result_img', fontsize=20)

plt.show()
'''