# Visual inspection of motorcycle connecting rods.

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

# NOTE First task images: 00, 01, 12, 21, 31, 33
# NOTE Second task images:
#   1. 44, 47, 48, 49
#   2. 50, 51
#   3. 90, 92, 98

basic_images = ('00', '01', '12', '21', '31', '33')
distr_images = ('44', '47', '48', '49')
conta_images = ('50', '51')
powde_images = ('90', '92', '98')

# Path & names
folder = os.path.join('assets', 'rods')
img_prefix = 'TESI'
img_extens = '.BMP'

# plot dicts
plot_imgs = []
plot_titl = []
plot_hist = []

plot_cols = 0

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

# Constants

# colors
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
RED    = (255, 0, 0)
GREEN  = (0, 255, 0)
BLUE   = (0, 0, 255)
L_BLUE = (3, 142, 170) # light blue
ORANGE = (234, 114, 2)
PURPLE = (234, 2, 219)
CYAN   = (55, 220, 229)

# Typos
font          = cv2.FONT_HERSHEY_PLAIN
txt_scale     = 2
txt_thickness = 2
txt_line_type = cv2.LINE_8

# opencv hierarchy indexes
NEXT     = 0
PREVIOUS = 1
F_CHILD  = 2
PARENT   = 3

X = 0
Y = 1

####################
# HELPER FUNCTIONS
####################

def log_image_infos(image, image_name=""):
    src_dim = image_name + " HxW: " + str(image.shape[0]) + " x " + str(image.shape[1])
    bit_dpt = image_name + " bit depth: " + str(image.dtype)

    logging.info(src_dim)
    logging.info(bit_dpt)


def evaluate_histogram(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    return hist


def plot_image(image, title="", eval_hist=False):
    global plot_cols
    
    plot_imgs.append(image)
    plot_titl.append(title)

    plot_cols += 1
    
    hist = None
    if eval_hist:
        hist = evaluate_histogram(image)

    plot_hist.append(hist)


def show_images(columns=1, rows_per_sheet=3):
    max_plots = columns * rows_per_sheet # max number of plots per sheet
    tot_imgs = len(plot_imgs)

    rows = tot_imgs // columns
    figures = 1

    if rows > rows_per_sheet:
        logging.warning("Too much rows to plot in a single page. To avoid overlapping, images will be splitted in different pages.")
        figures = math.ceil(rows / rows_per_sheet)
        rows = rows_per_sheet

    for f in range(figures):
        plt.figure(f)
        base_ind = max_plots * f
        maxi = max_plots
        if f == figures - 1:
            maxi = tot_imgs - ((figures - 1) * max_plots)
        for i in range(maxi):
            plt.subplot(rows, columns, i+1)
            plt.title(plot_titl[base_ind + i])
            plt.imshow(plot_imgs[base_ind + i], cmap='gray')
            plt.axis('off')

    legend = []
    for h in range(tot_imgs):
        if plot_hist[h] is not None:
            plt.figure(figures)
            plt.title("Histogram of ")
            plt.plot(plot_hist[h])
            legend.append(plot_titl[h])
    plt.legend(legend, loc = 'upper left')

    plt.show()


def lerp(start, end, t):
    """
    1D linear interpolation
    """
    return (1 - t) * start + t * end


def euclidean_length(a, b):
    return math.sqrt(a**2 + b**2)


def find_rods_connections(contour):
    """
    Evaluate connections through convexity defects.
    """

    # returns a structure containing start_point, end_point, depth_point, depth
    conv_defects = cv2.convexityDefects( contour, cv2.convexHull(contour, returnPoints = False) )
    
    # pruning subsequent points
    threshold = 10

    defects = np.sort(conv_defects[:, 0, 2])
    max_i = len(defects)

    prune = []

    for curr_i in range(len(defects) - 1):
        next_i = curr_i + 1

        if defects[next_i] - defects[curr_i] <= threshold:
            prune.append(defects[curr_i])
            prune.append(defects[next_i])

    # evaluate first-last index distances
    first = 0
    last = max_i - 1

    if (len(contour) - defects[last]) + defects[first] <= threshold:
        prune.append(defects[first])
        prune.append(defects[last])

    for pr in prune:
        defects = np.delete(defects, np.argwhere(defects == pr))

    # re-evaluate length
    max_i = len(defects)

    # evaluate contact points
    contact_pts = []

    shift_i = 10

    for curr_i in defects:
        prev_i = curr_i - shift_i
        next_i = curr_i + shift_i

        if next_i > max_i:
            next_i = (curr_i + shift_i) - max_i
        #if prev_i < 0:
        #    pass # python should reverse automatically

        curr_pt = contour[curr_i][0]
        prev_pt = contour[prev_i][0]
        next_pt = contour[next_i][0]
        
        diff1 = ( curr_pt[X] - prev_pt[X], curr_pt[Y] - prev_pt[Y] )
        diff2 = ( next_pt[X] - curr_pt[X], next_pt[Y] - curr_pt[Y] )

        if( (diff1[X] < 0 and diff2[X] > 0) or (diff1[X] > 0 and diff2[X] < 0) or
            (diff1[Y] < 0 and diff2[Y] > 0) or (diff1[Y] > 0 and diff2[Y] < 0) ):
            contact_pts.append(curr_pt)

    return contact_pts


def evaluate_line(start, end):
    diff = (end[X] - start[X], end[Y] - start[Y])
    
    increment = 0.01
    t = 0

    line = []

    curr_x = int(round(start[X]))
    curr_y = int(round(start[Y]))

    line.append( (curr_x, curr_y) )

    while t < 1:
        t += increment

        p_x = int(round( lerp(start[X], end[X], t) ))
        p_y = int(round( lerp(start[Y], end[Y], t) ))

        if p_x != curr_x or p_y != curr_y: # new pixel found
            curr_x = p_x
            curr_y = p_y

            line.append( (curr_x, curr_y) )

    return line


def evaluate_bezier_curve(cpts):
    """ Evaluate discrete bezier curve from given control points using the De Casteljau uniform algorithm.
    """
    
    size = len(cpts)
    n = size - 1 # curve's degree

    t_increment = 0.01
    t = 0

    bezier_curve = []
    matrix = np.zeros( (size, size, 2) ) # tmp xy matrix

    while t < 1:
        # copy the cpts values in a matrix we can work on
        for j in range(size):
            matrix[0][j][X] = cpts[j][X]
            matrix[0][j][Y] = cpts[j][Y]

        # calculate the curve
        for i in range(1, size):
            for j in range(0, size - i):
                matrix[i][j][X] = lerp(matrix[i-1][j][X], matrix[i-1][j+1][X], t)
                matrix[i][j][Y] = lerp(matrix[i-1][j][Y], matrix[i-1][j+1][Y], t)

        bezier_curve.append( ( int(round(matrix[n][0][X])), int(round(matrix[n][0][Y])) ) )
        t += t_increment

    return bezier_curve


def fix_connections(mask, contour):
    contact_pts = find_rods_connections(contour)
    
    while len(contact_pts) > 0:
        start = contact_pts[0]
        end   = contact_pts[1]

        contact_pts.remove(start)
        
        direction = (end[X] - start[X], end[Y] - start[Y])
        distance = euclidean_length(direction[X], direction[Y])
        
        for cpt in contact_pts:
            direct = (cpt[X] - start[X], cpt[Y] - start[Y])
            length = euclidean_length(direct[X], direct[Y])
            
            if length < distance:
                distance = length
                direction = direct
                end = cpt

        contact_pts.remove(end)

        middle = ( start[X] + (direction[X] / 2), start[Y] + (direction[Y] / 2) )

        direct = (direction[X] / distance, direction[Y] / distance) # normalize

        # calculate the perpendicular line as the cross product between major axis and the z axis (out from display)
        perp1 = np.cross(direct, (0, 0,  1))
        perp2 = np.cross(direct, (0, 0, -1))

        # move perpendicularly from middle point by a given factor
        factor = 5.5
        perp1 = (middle[X] + (perp1[X] * factor), middle[Y] + (perp1[Y] * factor))
        perp2 = (middle[X] + (perp2[X] * factor), middle[Y] + (perp2[Y] * factor))
        
        line = evaluate_line(middle, perp1)

        # if neighbour of a background pixel,
        # then set opposite perpendicular point as middle control point
        for px in line:
            if mask[px[Y]][px[X]] == 0:
                middle = perp2
                break
        
        if middle != perp2:
            line = evaluate_line(middle, perp2)
            for px in line:
                if mask[px[Y]][px[X]] == 0:
                    middle = perp1
                    break
        
        if middle == perp1 or middle == perp2: # draw curve
            # control points of the touching surface
            cpts = np.array([ 
                [start[X], start[Y]], 
                [int(round(middle[X])), int(round(middle[Y]))],
                [end[X], end[Y]] 
            ])
            
            bezier_pts = evaluate_bezier_curve(cpts)
            for bpt_i in range(len(bezier_pts) - 1):
                start = bezier_pts[bpt_i]
                end   = bezier_pts[bpt_i + 1]

                cv2.line(mask, start, end, BLACK, 2)
        else: # draw straight line
            cv2.line(mask, (start[X], start[Y]), (end[X], end[Y]), BLACK, 2)

    return mask


def evaluate_blobs(binary_mask):
    mask, contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    rod_count = 1
    rods = []
    for i in range(hierarchy.shape[1]): # loop through hierarchy rows
        if hierarchy[0][i][PARENT] == -1 and hierarchy[0][i][F_CHILD] != -1: # 1. is external contour? 2. possible rod?
            # evaluate MER (minimum (oriented) enclosing rectangle)
            mer = cv2.minAreaRect(contours[i])
            
            logging.debug("MER " + str(mer)) # (center (x,y), (width, height), angle of rotation as in https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
            
            mer_centre = mer[0]
            dims = (
                mer[1][0] if mer[1][0] < mer[1][1] else mer[1][1], # width < height
                mer[1][1] if mer[1][1] > mer[1][0] else mer[1][0], # height > width
            )
            rot_angle = mer[2]

            elong = dims[1] / dims[0] # elognatedness

            if elong > 1.5: # if elongated more than a threshold (0.5) then it is a rod
                if len(contours[i]) > 450: # if connected rods
                    fix_connections(mask, contours[i])

                    plot_image(mask, "Fixed rods connections")

                    rods = evaluate_blobs(mask)

                    break
                else:
                    ch1 = hierarchy[0][i][F_CHILD] # first hole index
                    ch2 = hierarchy[0][ch1][NEXT]  # second hole index

                    # classify rod and  evaluate holes
                    rod_name = "A" + str(rod_count)
                    circle1 = cv2.fitEllipse(contours[ch1]) # returns rotated rectangle (see MER)
                    circle2 = None
                    if ch2 != -1: # type B
                        rod_name = "B" + str(rod_count)
                        circle2 = cv2.fitEllipse(contours[ch2])
                    rod_count += 1 # keep track of rods. Just for name them

                    # evaluate barycenter through scanline algorithm
                    y_min = np.min(contours[i][:, 0, Y])
                    y_max = np.max(contours[i][:, 0, Y])

                    tot_x = 0
                    tot_y = 0
                    area = 0

                    for y_px in range(y_min, y_max + 1):
                        bounds = contours[i][ np.where(contours[i][:, 0, Y] == y_px)[0] ][:, :, X] # external bounds (x axis)

                        bounds = bounds.reshape(1, -1) # reshape to 1D array

                        b_min = np.min(bounds[0])
                        b_max = np.max(bounds[0])

                        count = 0
                        for x_px in range(b_min, b_max + 1):
                            if np.all(mask[y_px][x_px] == WHITE):
                                count += 1
                                tot_x += x_px

                                # debug
                                #final[y_px][x_px] = BLUE

                        area += count
                        tot_y += y_px * count

                    barycenter = (0, 0)
                    if area > 0:
                        barycenter = np.array([tot_x / area, tot_y / area])
                    
                    # Width at barycenter
                    
                    # evaluate major axis as hole1's center - barycenter
                    major_axis = (circle1[0][X] - barycenter[X], circle1[0][Y] - barycenter[Y], 0) # 3D needed for cross product

                    # normalize
                    length = euclidean_length(major_axis[X], major_axis[Y])
                    major_axis = (major_axis[X] / length, major_axis[Y] / length)

                    # calculate the perpendicular line as the cross product between major axis and the z axis (out from display)
                    perp = np.cross(major_axis, (0, 0, 1))

                    # move perpendicularly from barycenter by a given factor
                    factor = 50
                    perp = (barycenter[X] + (perp[X] * factor), barycenter[Y] + (perp[Y] * factor))

                    t = 0
                    increment = 0.1 / math.sqrt(perp[X]**2 + perp[Y]**2)
                    diagonal = math.sqrt(2)

                    curr_x = int(round(barycenter[X]))
                    curr_y = int(round(barycenter[Y]))
                    
                    wab = 0 # width at barycenter
                    while t < 1:
                        t += increment

                        p_x = int(round( lerp(barycenter[X], perp[X], t) ))
                        p_y = int(round( lerp(barycenter[Y], perp[Y], t) ))

                        if p_x != curr_x or p_y != curr_y: # new pixel found
                            if mask[curr_y][curr_x] == 255: # stop if background encountered
                                
                                # shift along x or y then increment by 1 
                                # diagonal movement then increment by sqrt(2)
                                wab += diagonal if p_x != curr_x and p_y != curr_y else 1

                                curr_x = p_x
                                curr_y = p_y

                                # debug
                                #final[curr_y][curr_x] = RED
                            else:
                                logging.debug("Break at t=" + str(t))
                                break

                    logging.debug("Final t is " + str(t))
                    wab *= 2 # both sides of the barycenter

                    # location where to put rod name
                    text_location = (contours[i][0, 0, X], contours[i][0, 0, Y])

                    rods.append( (mer, barycenter, circle1, circle2, rod_name, text_location, dims, wab) )

    return rods


############
# Main
############
logging.info("OpenCV version is " + cv2.__version__)

""" basic_images = ('00', '01', '12', '21', '31', '33')
distr_images = ('44', '47', '48', '49')
conta_images = ('50', '51')
powde_images = ('90', '92', '98') """

for image_i in conta_images:
    plot_cols = 0

    image_name = img_prefix + image_i + img_extens
    source = cv2.imread(os.path.join(folder, image_name), cv2.IMREAD_GRAYSCALE)
    plot_image(source, "Original - " + image_i, eval_hist=True)

    log_image_infos(source, image_name)

    #################
    # Denoising
    #################

    filtered = cv2.medianBlur(source, ksize=3)
    plot_image(filtered, "Denoised")

    histogram = evaluate_histogram(filtered)

    #################
    # Binarization
    #################

    # Inverted because in OpenCV contour detection need white object over black background. See https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    ret, binarized = cv2.threshold(filtered , 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plot_image(binarized, "Binarized")

    ####################
    # Binary Morphology
    ####################

    #kernel = np.ones((3,3),np.uint8)

    #closing = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    #plot_image(closing, "Closing")

    #####################
    # Final calculations
    #####################

    final = source.copy() # apply draws on source copy
    final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    # debug
    #cv2.drawContours(final, contours, -1, RED, 2)

    for rod in evaluate_blobs(binarized):
        mer = rod[0]
        barycenter = rod[1]
        circle1 = rod[2]
        circle2 = rod[3]
        rod_name = rod[4]
        text_location = rod[5]
        dims = rod[6]
        wab = rod[7]

        # prepare MER box to be drawn
        box = cv2.boxPoints(mer)
        box = np.int0(box)

        # draw barycenter
        #cv2.circle(final, (round(mer[0][0]), round(mer[0][1])), 1, L_BLUE, 2) # MER (not exact)
        cv2.circle(final, (int(round(barycenter[0])), int(round(barycenter[1]))), 1, RED, 2) # scanline (exact)
        
        # draw holes and report their infos
        cv2.ellipse(final, circle1, ORANGE, 2)
        cv2.circle(final, (round(circle1[0][0]), round(circle1[0][1])), 1, ORANGE, 2)
        hole_info = ("- Hole 1 (orange) -" + "\n" +
            "Location x: " + str(circle1[0][0]) + "\n" +
            "Location y: " + str(circle1[0][1]) + "\n" +
            "Diameter:   " + str((circle1[1][0] + circle1[1][1]) / 2) # not perfect circle. So W+H / 2
        )
        if circle2 is not None:
            cv2.ellipse(final, circle2, PURPLE, 2)
            cv2.circle(final, (round(circle2[0][0]), round(circle2[0][1])), 1, PURPLE, 2)
            hole_info += ("\n- Hole 2 (purple) -" + "\n" +
            "Location x: " + str(circle2[0][0]) + "\n" +
            "Location y: " + str(circle2[0][1]) + "\n" +
            "Diameter:   " + str((circle2[1][0] + circle2[1][1]) / 2) # not perfect circle. So W+H / 2
            )
        
        # draw MER
        cv2.drawContours(final, [box], 0, GREEN, 1)

        # draw numbered rod type
        cv2.putText(
            final, 
            rod_name, 
            text_location, 
            font, 
            txt_scale, 
            BLUE, 
            txt_thickness, 
            txt_line_type)
        
        # complete report
        print("\n====== Rod " + rod_name + " in image " + image_i + " ======" + "\n" +
            "Location x:  " + str(mer[0][0])  + "\n" +
            "Location y:  " + str(mer[0][1])  + "\n" +
            "Orientation: " + str(mer[2])     + "\n" +
            "Length:      " + str(dims[1])    + "\n" +
            "Width:       " + str(dims[0])    + "\n" +
            "Barycenter:  " + str(barycenter) + "\n" +
            "WaB:         " + str(wab)        + "\n" +
            hole_info
        )

    plot_image(final, "Final")

logging.debug("Total columns to plot: " + str(plot_cols))

show_images(columns=plot_cols, rows_per_sheet=2)