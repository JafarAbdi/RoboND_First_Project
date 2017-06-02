import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 255
    # Return the binary image
    return color_select

# Function for detecting rocks input: image, output: rocks masked
def detect_rock(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of yellow color in HSV
    lower_yellow = np.array([80,100,100])
    upper_yellow = np.array([100,255,255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    return mask

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = (yaw*np.pi)/180
    # Apply a rotation
    xpix_rotated = np.cos(yaw_rad)*xpix - np.sin(yaw_rad)*ypix
    ypix_rotated = np.sin(yaw_rad)*xpix + np.sin(yaw_rad)*ypix
    # Return the result
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = xpos + xpix_rot/scale
    ypix_translated = ypos + ypix_rot/scale
    # Return the result
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img

    #some constants
    bottom_offset = 6
    dst_size = 5
    scale = 10
    #Add mask to limit the viewable area
    vertices = np.array([[(0,0),(Rover.img.shape[1],0),(Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset),(Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset)]],dtype=np.int32)
    mask = np.zeros_like(Rover.img[:,:,0])
    cv2.fillPoly(mask, vertices, 255)
    #kernel for morphologyEx function
    kernel = np.ones((5,5))
    # 1) Define source and destination points for perspective transform
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img,source,destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    #detect terrain
    terrain = color_thresh(warped)
    terrain = cv2.morphologyEx(terrain,cv2.MORPH_OPEN,kernel,iterations=1)
    #terrain = cv2.morphologyEx(terrain,cv2.MORPH_CLOSE,kernel,iterations=1)
    #detect obstacles
    obstacle = cv2.dilate(terrain,kernel,iterations=1) 
    obstacle = cv2.bitwise_and(obstacle,cv2.bitwise_not(terrain))
    #detect rocks
    rock = detect_rock(warped)

    #mask terrain and obstacles
    obstacle = cv2.bitwise_and(mask,obstacle)
    terrain  = cv2.bitwise_and(mask,terrain)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = obstacle
    Rover.vision_image[:,:,1] = rock
    Rover.vision_image[:,:,2] = terrain

    # 5) Convert map image pixel values to rover-centric coords
    xpix_o,ypix_o = rover_coords(Rover.vision_image[:,:,0])
    xpix_r,ypix_r = rover_coords(Rover.vision_image[:,:,1])
    xpix_t,ypix_t = rover_coords(Rover.vision_image[:,:,2])
    # 6) Convert rover-centric pixel values to world coordinates
    x_pix_o_world,y_pix_o_world = pix_to_world(xpix_o,ypix_o,Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0],10)
    x_pix_r_world,y_pix_r_world = pix_to_world(xpix_r,ypix_r,Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0],10)
    x_pix_t_world,y_pix_t_world = pix_to_world(xpix_t,ypix_t,Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0],10)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    if Rover.pitch <= 0.75 or Rover.pitch >= 359.25:
        if Rover.roll <= 1 or Rover.roll >= 359:
            Rover.worldmap[y_pix_t_world, x_pix_t_world, 0] += 1
            Rover.worldmap[y_pix_r_world, x_pix_r_world, 1] += 1
            Rover.worldmap[y_pix_o_world, x_pix_o_world, 2] += 1
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpix_t,ypix_t)
    
 
    
    
    return Rover