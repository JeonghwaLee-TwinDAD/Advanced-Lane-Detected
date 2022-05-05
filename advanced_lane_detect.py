import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import glob, os
from moviepy.editor import VideoFileClip
import line_info as lane_fn

xm_per_pix = 3.7 / 600 # lane width is 3.7 meters  
ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
img_size = [720, 1280]       

# define a class
class Line():                     
    def __init__(self):       
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters to the line from x = 0 
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        # restart search with initial condition         

    def smoothing(self):
        n_fitted = 10 #Iteration number of fitted *_fit variable
        if len(self.current_fit) > n_fitted or len(self.recent_xfitted) > n_fitted:
            # throw out old fits, keep newest n
            self.current_fit = self.current_fit[1:len(self.current_fit)]
            self.recent_xfitted = self.recent_xfitted[1:len(self.recent_xfitted)]
            self.best_fit = np.average(self.current_fit, axis=0)
        
        if len(self.current_fit) == 0:
            self.best_fit = None
        else:
            self.best_fit = np.average(self.current_fit, axis=0)

    # update stored data when lanes were detected and fitted
    def update(self, fit, fit_x_int, x, y):    
        # add lane polynomial coefficients and smoothing for the most recent fit
        self.current_fit.append(fit)
        # add x values of the last n fits of the lines
        self.recent_xfitted.append(fit_x_int)         
        # best fit after smoothing the lines
        self.best_fit = np.average(self.current_fit, axis= 0) 
        # best n-fitted polynominal curve
        self.bestx = np.average(self.recent_xfitted, axis=0)         
        # distance in meters to vehicle center from x = 0
        self.line_base_pos = self.bestx * xm_per_pix           
        #difference in fit coefficients between last and new fits
        self.diffs = abs(fit - self.best_fit)
        #x y values for detected line pixels
        self.allx = x 
        self.ally = y
        self.radius_of_curvature = lane_fn.curv_rad(img_size, self.allx, self.ally, xm_per_pix, ym_per_pix)
        # find the curvatures at lanes each and offset of ceter distance (unit : meter) 

    def sanity_check(self, x_int_diff):
        # Check if the base ditance 350 < x_diff < 700
        if (x_int_diff > 350) and (x_int_diff < 700):
            self.detected = True
        else:
            self.detected = False
        '''
        # Check if fit difference is proper (diffs) 
        if (self.diffs[0] > 0.001 or self.diffs[1] > 1.0 or self.diffs[2] > 100):
            self.detected = False
        else:
            self.detected = True
        '''
def process_image(input_img):
    new_img = np.copy(input_img)
    img_size = new_img.shape
    
    # Contrast correction of initial images to fight excessive darkness or brightness (not affective)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2YUV)
    new_img[:,:,0] = cv2.equalizeHist(new_img[:,:,0])
    new_img = cv2.cvtColor(new_img, cv2.COLOR_YUV2RGB)
    
    # pipeline process to find binary image
    warped_bin, Minv = lane_fn.pipeline(new_img)
            
    ### Apply for the Tips and Tricks
    ## Check if best_fit is at both left and right lane (reset condition #1)
    # if both left and right lines were detected last frame, use search around poly(), otherwise use sliding window
    if l_line.detected is False or r_line.detected is False:
        window_bin, left_fit, right_fit, leftx, lefty, rightx, righty = lane_fn.window_search(warped_bin)
    else:
        window_bin, left_fit, right_fit, leftx, lefty, rightx, righty = lane_fn.poly_search(warped_bin, l_line.best_fit, r_line.best_fit)
    # end if (checking best_fit is None or not )

    ## Take initial line point at x-axis
    if left_fit is not None :
        l_fit_x_int = left_fit[0]*img_size[0]**2 + left_fit[1]*img_size[0] + left_fit[2]
    else:
        l_fit_x_int = 0
        
    if right_fit is not None:
        r_fit_x_int = right_fit[0]*img_size[0]**2 + right_fit[1]*img_size[0] + right_fit[2]
    else:
        r_fit_x_int = img_size[1] # at the end of x

    ## check if left and right lane detacted 
    x_int_diff = abs(r_fit_x_int - l_fit_x_int)    
    l_line.sanity_check(x_int_diff)
    r_line.sanity_check(x_int_diff)

    # Update stored data
    if l_line.detected is True:
        l_line.update(left_fit, l_fit_x_int, leftx, lefty)        
    else:
        l_line.current_fit = l_line.current_fit[1:len(l_line.current_fit)]
        l_line.recent_xfitted = l_line.recent_xfitted[1:len(l_line.recent_xfitted)]

    if r_line.detected is True:
        r_line.update(right_fit, r_fit_x_int, rightx, righty)
    else:
        r_line.current_fit = r_line.current_fit[1:len(r_line.current_fit)]
        r_line.recent_xfitted = r_line.recent_xfitted[1:len(r_line.recent_xfitted)]
    
    ## define Best fit and the number of current fit  
    l_line.smoothing()
    r_line.smoothing()
           
    # Check if best fit is to draw lanes
    if l_line.best_fit is not None and r_line.best_fit is not None:
        # lane width is 3.7 meters
        xm_per_pix = 3.7 / (r_line.line_base_pos - l_line.line_base_pos)
        # Draw the lane on the road
        polygon = lane_fn.draw_lane(new_img, warped_bin, l_line.best_fit, r_line.best_fit, Minv)
        # Draw the radius of curvature and offset
        car_position = img_size[1]/2 * xm_per_pix                
        lane_center_position = (r_line.line_base_pos + l_line.line_base_pos)/2        
        center_dist = car_position - lane_center_position 
        final_image = lane_fn.draw_data(polygon,(l_line.radius_of_curvature+r_line.radius_of_curvature)/2, center_dist)
    else:
        print("Can not find best fit curves!")
        final_image = input_img

    # Diagonstic process
    final_image = lane_fn.diagonstic(final_image, warped_bin, l_line , r_line, Minv)

    return final_image

l_line = Line() #Line class instance left (global variable)
r_line = Line() #Line class instance right (global variable)  


# Draw the images
img= cv2.imread('./test_images/test1.jpg')
#img= cv2.imread('./challenge_images/challenge_center.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#top_down, Mivn = lane_fn.pipeline(img)
#result_img, left_fit, right_fit, leftx, lefty, rightx, righty = lane_fn.window_search(top_down)        

result_img = process_image(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
f.tight_layout()
ax1.imshow(img, cmap='gray')
ax1.axis('off')
ax1.set_title('original', fontsize=20)
ax2.imshow(result_img, cmap='gray')
ax2.axis('off')
ax2.set_title('result_img', fontsize=20)
plt.show()

'''
# Create video clip
video_input = VideoFileClip('./video_clip/harder_challenge_video.mp4')#.subclip(0,5)
video_output = './video_clip/harder_challenge_video_output.mp4'
processed_video = video_input.fl_image(process_image)
processed_video.write_videofile(video_output, audio=False)



# Create camera calibration data (dist_pickle dictionary)
images = glob.glob('./camera_cal/calibration*.jpg')
# Find cal data with 6*9 chessboard calibration at many images
objpoints, imgpoints = lane_fn.cal_camera(images, 6, 9)

image = cv2.imread('./camera_cal/calibration1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

undistorted, mtx, dist = lane_fn.cal_undistort(image, objpoints, imgpoints)

dist_pickle = {}
dist_pickle["objpoints"] = objpoints  
dist_pickle["imgpoints"] = imgpoints
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open( "distortion_pickle.p", "wb" ) )


# RIO check routine
img= cv2.imread('./challenge_images/challenge_initial.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

warped, Minv = lane_fn.warp(img)
lane_fn.region_of_interest(img, warped)
plt.show()
'''

