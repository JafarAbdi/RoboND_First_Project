# RoboND_First_Project

Notebook Analysis

1- to identify the rocks I writed a function called detect_rock() I converted the image to HSV format after that run inRange function to detect the color that I specify (Here yellow)
2- to detect the obstacles I had two options 
   1- by calling cv2.bitwise_not which will select the color that are not terrain 
   2- by using http://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html function (this method was recommended by a fellow in slack channel)
   in my code I used the second method
---------------------------------------------
1- I did the steps that mentioned in process_image function

Autonomous Navigation and Mapping

1- Video -> https://www.youtube.com/watch?v=gOOxzLtntbM
   FPS between 16-20
   Resultion 640x480

I added a detect_rock function and filled rotate_pix and translate_pix

For perception_step function 

-I difeined some constant bottom_offset, scale factor 10cm/m(pixle), size of destination square 2*5
-Add mask to limit the terrain and obstacles
-Define source and destination points
-Apply perspect_transform to the input image
-Identify the terrain and rocks, obstacles and mask them
-Add thersholded images to Rover.vision_image
-Convert map image pixel values to rover-centric coordinates after that to the world coordinates
-Update Rover.worldmap if the Rover.pitch <= 0.75 or Rover.pitch >= 359.25 and Rover.roll <= 1 or Rover.roll >= 359 the values 
selected experimentally
-Convert rover-centric pixel positions to polar coordinates by applying to_polar_coords to terrain pixles

I didn't do any change to decision_step function (I'll do when I start playing with Udacity challenge)
