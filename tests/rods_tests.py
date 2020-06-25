# test rods

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging, sys, os
import math

# vars
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED    = (255, 0, 0)
GREEN  = (0, 255, 0)
BLUE   = (0, 0, 255)
L_BLUE = (3, 142, 170) # light blue
ORANGE = (234, 114, 2)
PURPLE = (234, 2, 219)

font = cv2.FONT_HERSHEY_PLAIN
scale = 2
thickness = 2
line_type = cv2.LINE_8

NEXT     = 0
PREVIOUS = 1
F_CHILD  = 2
PARENT   = 3

X = 0
Y = 1

h_img = 10
w_img = 20

h_obj = 6
w_obj = 12

pos_x = 2
pos_y = 2

hol1_x = 4
hol1_y = 4
hol1_d = 3

#------------------
# create test image

image = np.zeros([h_img, w_img], dtype=np.uint8)

image = cv2.rectangle(image, (pos_x, pos_y), (pos_x + w_obj, pos_y + h_obj), WHITE, -1)
image = cv2.rectangle(image, (hol1_y, hol1_x), (hol1_y + hol1_d, hol1_x + hol1_d), BLACK, -1)

final = image.copy()
final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

#------------------
# contour detection

image, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#------------------
# rod detection

rod_count = 1
for i in range(hierarchy.shape[1]): # loop through hierarchy rows
    if hierarchy[0][i][PARENT] == -1 and hierarchy[0][i][F_CHILD] != -1: # 1. is external contour? 2. possible rod?
        # evaluate MER (minimum (oriented) enclosing rectangle)
        mer = cv2.minAreaRect(contours[i])
        #print(mer) # (center (x,y), (width, height), angle of rotation as in https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
        mer_centre = mer[0]
        dims = (
            mer[1][0] if mer[1][0] < mer[1][1] else mer[1][1], # width < height
            mer[1][1] if mer[1][1] > mer[1][0] else mer[1][0], # height > width
        )
        rot_angle = mer[2]

        elong = dims[1] / dims[0]

        if elong > 1.5: # if elongated more than a threshold (0.5) then it is a rod
            ch1 = hierarchy[0][i][F_CHILD]
            ch2 = hierarchy[0][ch1][NEXT]

            # classify rod and  evaluate holes
            rod_name = "A" + str(rod_count)
            circle1 = cv2.fitEllipse(contours[ch1]) # returns rotated rectangle (see MER)
            circle2 = None
            if ch2 != -1: # type B
                rod_name = "B" + str(rod_count)
                circle2 = cv2.fitEllipse(contours[ch2])
            rod_count += 1 # keep track of rods. Just for name them

            # evaluate barycentre TODO
            
            y_max = np.max(contours[i][:, 0, Y])
            y_min = np.min(contours[i][:, 0, Y])

            tot_x = 0
            tot_y = 0
            area = 0

            for y_px in range(y_min, y_max + 1):
                bounds = contours[i][ np.where(contours[i][:, 0, Y] == y_px)[0] ][:, :, X] # external bounds

                print("BOUNDS per y " + str(y_px) + "\n" + str(bounds))

                bounds = bounds.reshape(1, -1)

                print("RESHAPE " + str(bounds))

                b_min = np.min(bounds[0])
                b_max = np.max(bounds[0])

                print("MINMAX " + str(b_min) + " " + str(b_max))

                count = 0
                
                for x_px in range(b_min, b_max + 1):
                    if np.all(final[y_px][x_px] == WHITE):
                        count += 1
                        tot_x += x_px
                        final[y_px][x_px] = RED

                area += count
                tot_y += y_px * count

#------------
# show image
plt.subplot(2, 1, 1)
plt.imshow(image, cmap='gray')
plt.subplot(2, 1, 2)
plt.imshow(final)
plt.show()