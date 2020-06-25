# Development of an Augmented Reality System

# To execute install:
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
RED    = (255, 0, 0)
GREEN  = (0, 255, 0)
BLUE   = (0, 0, 255)
L_BLUE = (3, 142, 170) # light blue
ORANGE = (234, 114, 2)
PURPLE = (234, 2, 219)

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


def play_video(path, name):
    cap = cv2.VideoCapture(os.path.join(path, name))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            cap.release()
            print("Video resource released.")
            break

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

        cap.release()
        cv2.destroyAllWindows()


def save_video(video, path, name, width, height, fps):
    # TODO
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(os.path.join(path, name), fourcc, fps, (width, height))

    while video.isOpened():
        ret, frame = video.read()
        if not ret or frame is None:
            logging.warning("No frame received (stream end?). Exiting...")
            break
        out.write(frame)

    out.release()


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

plot_image(reference, "Reference Frame")

# Load AR layer image
ar_layer = cv2.imread(os.path.join(folder, ar_layer_name), cv2.IMREAD_COLOR)
ar_layer_mask = cv2.imread(os.path.join(folder, ar_layer_mask_name), cv2.IMREAD_GRAYSCALE)

layer = obj_mask_extraction(ar_layer, ar_layer_mask)

# TODO Devi ritagliare la label in due parti corrispndenti alle due etichette e sicronizzarle..........
# oppure ritagli l'immagine e tieni la maschera e la riusi così com'è.

height_ref = reference.shape[0]
width_ref  = reference.shape[1]

height_label = layer.shape[0]
width_label  = layer.shape[1]

# the width of the layer is >> than the reference. We crop it
layer = layer[:, 200:500]

height_label = layer.shape[0]
width_label  = layer.shape[1]

plot_image(layer, "AR layer")

# Load input video
src_video = cv2.VideoCapture(os.path.join(folder, source_name))

current_frame = 0
# move frame by frame
while src_video.isOpened():
    if cv2.waitKey(1) == ord('q'):
        logging.info("Exiting...")
        src_video.release()
        cv2.destroyAllWindows()
        sys.exit()

    ret, frame = src_video.read()

    if not ret or frame is None:
        src_video.release()
        print("Input video resource released.")
        break

    current_frame += 1
    plot_image(frame, "Current frame (" + str(current_frame) + ")")

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
    threshold    = 0.7

    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    # Object found?
    if len(good_matches) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Getting the coordinates of the corners of our query object in the train image
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        print("M1 " + str(M))

        # get current frame dimensions
        height_frame = frame.shape[0]
        width_frame  = frame.shape[1]

        print("REF " + str(width_ref) + "-" + str(height_ref) + "\n" +
        "LABEL " + str(width_label) + "-" + str(height_label) + "\n" +
        "FRAME " + str(width_frame) + "-" + str(height_frame) + "\n" )

        ref_pts = np.float32([
            [0, 0],
            [0, height_ref - 1],
            [width_ref - 1, height_ref - 1],
            [width_ref - 1, 0]]).reshape(-1, 1, 2)
        
        ref_dst = cv2.perspectiveTransform(ref_pts, M)

        # Getting the homography to project ar layer on the surface of the query object.
        pts_label = np.float32([
            [0, 0],
            [0, height_label - 1],
            [width_label - 1, height_label - 1],
            [width_label - 1, 0]]).reshape(-1, 1, 2)

        print("REF_DST " + str(ref_dst))

        M = cv2.getPerspectiveTransform(pts_label, ref_dst)
        print("M2 " + str(M))

        #plot_matches(reference, kp_ref, frame, kp_frame, good_matches, matches_mask)

        # Warping the ar layer
        warped = cv2.warpPerspective(layer, M, (width_frame, height_frame))

        # Warp a white mask to understand what are the black pixels
        white = np.ones([height_label, width_label], dtype=np.uint8) * 255
        warp_mask = cv2.warpPerspective(white, M, (width_frame, height_frame))

        plt.figure(49)
        plt.imshow(warped)
        plt.show()

        # Restore previous values of the train images where the mask is black
        warp_mask = np.equal(warp_mask, 0)
        warped[warp_mask] = frame[warp_mask]

        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
        # Displaying the result
        plt.figure(49)
        plt.imshow(warped)
        plt.show()
        src_video.release()
        cv2.destroyAllWindows()
    else:
        msg = "Not enough matches are found - " + str(len(good_matches)) + "/" + str(MIN_MATCH_COUNT)
        logging.info(msg)
        matchesMask = None
