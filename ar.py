# Development of an Augmented Reality System

# To execute, install:
# pip3 install numpy
# pip3 install matplotlib
# pip3 install opencv-python==3.4.2.16
# pip3 install opencv-contrib-python==3.4.2.16

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging, sys, os
import math

# Path & names
folder = os.path.join('assets', 'ar')
source_name = 'Multiple View.avi'
final_name = 'Augmented Multiple View.avi'
ref_frame_name = 'ReferenceFrame.png'
ref_frame_mask_name = 'ObjectMask.PNG'
ar_layer_name = 'AugmentedLayer.png'
ar_layer_mask_name = 'AugmentedLayerMask.png'

# Logging
FORMAT = "%(asctime)-15s -- %(levelname)-9s: %(message)s"
logging.basicConfig(stream = sys.stdout, format = FORMAT, level = logging.INFO)

###############################
# REMINDER
# logging levels as for py3.7:
#
# Level    |  Numeric value
# ---------|----------------
# CRITICAL |    50
# ERROR    |    40
# WARNING  |    30
# INFO     |    20
# DEBUG    |    10
# NOTSET   |     0
###############################

figure = 1

MIN_MATCH_COUNT = 4

# Colors
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
RED    = (255, 0, 0)
GREEN  = (0, 255, 0)
BLUE   = (0, 0, 255)
L_BLUE = (3, 142, 170) # light blue
ORANGE = (234, 114, 2)
PURPLE = (234, 2, 219)

X = 0
Y = 1

####################
# HELPER FUNCTIONS
####################

def log_video_infos(video, video_name=""):
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)
    length = video.get(cv2.CAP_PROP_FRAME_COUNT)

    src_dim = video_name + " HxW:    " + str(height) + " x " + str(width)
    src_fps = video_name + " fps:    " + str(fps)
    src_cnt = video_name + " length: " + str(length)

    logging.info(src_dim)
    logging.info(src_fps)
    logging.info(src_cnt)


def plot_image(image, title=""):
    global figure

    plt.figure(figure)
    plt.title(title)
    plt.subplot(1, 1, 1)
    #plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    figure += 1


def plot_matches(img_query, kp_query, img_train, kp_train, good, matches_mask):
    plt.figure(50)
    draw_params = dict(matchColor = GREEN, singlePointColor = None, matchesMask = matches_mask, flags = 2)
    img = cv2.drawMatches(img_query, kp_query, img_train, kp_train, good, None, **draw_params)
    plt.imshow(img)


def obj_mask_extraction(image, mask):
    img = image.copy()
    img[np.logical_not(mask)] = np.asarray([255, 255, 255])
    
    return img


######################
# Main
######################

# Load reference frame
ref_frame = cv2.imread(os.path.join(folder, ref_frame_name), cv2.IMREAD_COLOR)
ref_mask  = cv2.imread(os.path.join(folder, ref_frame_mask_name), cv2.IMREAD_GRAYSCALE)

reference = obj_mask_extraction(ref_frame, ref_mask)

#debug
#plot_image(reference, "Reference Frame")

# Load AR layer image
ar_layer = cv2.imread(os.path.join(folder, ar_layer_name), cv2.IMREAD_COLOR)
ar_layer_mask = cv2.imread(os.path.join(folder, ar_layer_mask_name), cv2.IMREAD_GRAYSCALE)

layer = obj_mask_extraction(ar_layer, ar_layer_mask)

height_ref = reference.shape[0]
width_ref  = reference.shape[1]

height_label = layer.shape[0]
width_label  = layer.shape[1]

# the width of the layer is >> than the reference. We crop it
x_min = 200
x_max = 480
y_min = 20
y_max = 410
layer = layer[y_min:y_max, x_min:x_max]
ar_layer_mask = ar_layer_mask[y_min:y_max, x_min:x_max]

height_label = layer.shape[0]
width_label  = layer.shape[1]

#debug
#plot_image(layer, "AR layer")

# Load input video
src_video = cv2.VideoCapture(os.path.join(folder, source_name))

height_frame = int(src_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width_frame  = int(src_video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps          = src_video.get(cv2.CAP_PROP_FPS)
video_length = src_video.get(cv2.CAP_PROP_FRAME_COUNT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(os.path.join(folder, final_name), fourcc, fps, (width_frame, height_frame), isColor=True)

current_frame = 0
# move frame by frame
while src_video.isOpened():
    if cv2.waitKey(1) == ord('q'):
        logging.info("Exiting...")
        src_video.release()
        out.release()
        cv2.destroyAllWindows()
        sys.exit()

    ret, frame = src_video.read()

    if not ret or frame is None:
        logging.info("End reached.")
        break

    current_frame += 1

    # debug
    #plot_image(frame, "Current frame (" + str(current_frame) + ")")

    # Scale Invariant Feature Transform
    sift = cv2.xfeatures2d.SIFT_create() # sift detector initialization

    # keypoints SIFT detection on reference object and current frame
    kp_ref   = sift.detect(reference)
    kp_frame = sift.detect(frame)

    # compute SIFT descriptors
    kp_ref, des_ref     = sift.compute(reference, kp_ref)
    kp_frame, des_frame = sift.compute(frame, kp_frame)

    FLANN_INDEX_KDTREE = 1

    index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann   = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_ref, des_frame, k = 2)

    # Check distance between first and second closest elements
    good_matches = []
    threshold    = 0.8

    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    # Object found?
    if len(good_matches) >= MIN_MATCH_COUNT:
        #debug
        #cap = cv2.drawMatches(reference, kp_ref, frame, kp_frame, good_matches[:MIN_MATCH_COUNT], 0, flags=2)
        #plot_image(cap)

        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Getting the coordinates of the corners of our query object in the train image
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # debug
        #plot_matches(reference, kp_ref, frame, kp_frame, good_matches, matches_mask)

        ref_pts = np.float32([
            [194, 27], [190, 406],
            [476, 409], [459, 26] ]).reshape(-1, 1, 2)
        
        ref_dst = cv2.perspectiveTransform(ref_pts, M)

        #debug
        #img2 = cv2.polylines(frame, [np.int32(ref_dst)], True, GREEN, 3, cv2.LINE_AA)
        #plot_image(img2)

        # Getting the homography to project ar layer on the surface of the query object.
        pts_label = np.float32([
            [0, 0],
            [0, height_label - 1],
            [width_label - 1, height_label - 1],
            [width_label - 1, 0]]).reshape(-1, 1, 2)

        M = cv2.getPerspectiveTransform(pts_label, ref_dst)

        # Warping the ar layer
        warped = cv2.warpPerspective(layer, M, (width_frame, height_frame))

        # Warp a white mask to understand what are the black pixels
        warp_mask = cv2.warpPerspective(ar_layer_mask, M, (width_frame, height_frame))

        # Restore previous values of the train images where the mask is black
        warp_mask = np.equal(warp_mask, 0)
        warped[warp_mask] = frame[warp_mask]

        # debug
        #plt.figure(49)
        #plt.imshow(warp_mask, cmap='gray')
        #plt.figure(50)
        #plt.imshow(warped)
        #plt.show()
        #sys.exit()

        # save frame
        out.write(warped)
        logging.info("Frame " + str(current_frame) + "/" + str(int(video_length)) + " rendered.")
    else:
        msg = "Not enough matches are found - " + str(len(good_matches)) + "/" + str(MIN_MATCH_COUNT)
        logging.info(msg)
        matchesMask = None

src_video.release()
out.release()
cv2.destroyAllWindows()
logging.info("Video resources released.")