# RoboND_First_Project
---------------------------------------------

Notebook Analysis

1- to identify the rocks I writed a function called detect_rock() I converted the image to HSV format after that run inRange function to detect the color that I specify (Here yellow) I found the color range by runing the code in the end of this page http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#how-to-find-hsv-values-to-track

in this image it shown how the function detected the rock and it gives a good result

![Detect Rocks](https://github.com/JafarAbdi/RoboND_First_Project/blob/master/detect_rocks.png?raw=true)

---------------------------------------------

2- to detect the obstacles I had two options 

   1- by calling cv2.bitwise_not which will select the color that are not terrain 

   2- by using http://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html function (this method was recommended by a fellow in slack channel)
   in my code I used the second method

the steps used in detect obstacles function are as follow 

   1- detect the terrain area (left up corner image)

   2- dilating the terrain area with one iteration (right up corner)

   3- applying bitwise_not function to the original terrian (left down corner)

   4- taking bitwise_and between the dilated terrain and terrian not (right down corner)

![Detect Obstacles](https://github.com/JafarAbdi/RoboND_First_Project/blob/master/detect_obstacle.png?raw=true)


---------------------------------------------

1- for process_image function in notebook I did the same procedure for perception_step function in perception.py except rather than using Rover I used the Databucket instance data and the input image

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

-Update Rover.worldmap if the Rover.pitch <= 0.75 or Rover.pitch >= 359.25 and Rover.roll <= 1 or Rover.roll >= 359 the values selected experimentally

-Convert rover-centric pixel positions to polar coordinates by applying to_polar_coords to terrain pixles

I didn't do any change to decision_step function (I'll do when I start playing with Udacity challenge)

---------------------------------------------

I'm working now on udacity challenge 

to pick the samples I added 

if Rover.near_sample and not Rover.picking_up:

               # Set mode to "stop" and hit the brakes!

                Rover.throttle = 0
                
                # Set brake to stored brake value
                
                Rover.brake = Rover.brake_set
                
                Rover.steer = 0
                
                Rover.send_pickup = True
                
                Rover.samples_picked[np.argmin(Rover.dis_to_samples)] = 1


to decision_step function

and I'll add new variable to Rover class which store the magnitude of the distance from the rover to samples and pick them from min to max distance after that return to start position 
