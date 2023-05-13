import cv2
import numpy as np
import streamlit as st
from PIL import Image
from clearml import Task

# Vibrio recognition and counting Test Case, Problem 3
# - Preprocessing exploration: Identify vibrio color, Sharpening (Laplace kernel filter), Grayscale, Threshold otsu, Masking
# -- Find Plate to crop image using Hough Circle Transform
# --- Reason for use only cv library without machine learning model is performance consideration
# - Customizable Parameter, default accuracy fine tuned to the best config  
# - Draw circle boxes for vibrio detected 
# - Vibrio color segmentation and count [ongoing]
# - Implementation of Streamlit and ClearML

__author__ = 'Arrival Dwi Sentosa <arrivaldwisentosa@gmail.com>'
__source__ = ''

# streamlit
st.title('Problem 3: Vibrio recognition and counting')
st.write('Harap menunggu jika input file belum muncul, sedang initialization ClearML..')

# global variables
# color used to identify vibrio
houghColor = (0, 0, 255)
output_image = (lambda n, v: cv2.imwrite('outputs/' + n + '.png', v))
DEBUG = True

# clearml init
task = Task.init(project_name="ef-vibrio-count", task_name="vibrio")

# Hough Circle Transform parameter setting
sensitivity = st.slider('Sensitivity', 0, 10, 3)
nhood = st.slider('Neighborhood', 0, 30, 20)
param2 = st.slider('Accumulator', 1, 50, 20)

def pre_process(img_ori):
    # kernel to be used for strong laplace filtering
    kernel_strong = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]],
        dtype=np.float32)

    # kernel to be used for weak laplace filtering
    kernel_weak = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]],
        dtype=np.float32)

    # perform laplace filtering
    img_lap = cv2.filter2D(img_ori, cv2.CV_32F, kernel_weak)
    img_sharp = np.float32(img_ori) - img_lap
    if DEBUG:
        output_image("1. sharpened", img_sharp)

    # convert to 8bits gray scale
    img_sharp = np.clip(img_sharp, 0, 255).astype('uint8')
    img_gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        output_image("2. grayscale", img_gray)

    # binarize the greyscale image
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if DEBUG:
        output_image("3. binary", img_bin)

    # remove noise from the binary image
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((3, 3), dtype=int))
    if DEBUG:
        output_image("4. noise_reduction", img_bin)

    # find the circular plate mask in the image
    plate_mask, circle = find_plate(img_ori, img_bin)
    cv2.circle(img_ori, (int(circle[0]), int(circle[1])), int(circle[2]), houghColor, 2)

    img_ori = crop_image(img_ori, circle)
    
    inv = 0
    if np.sum(img_bin == 255) > np.sum(img_bin == 0):
        inv = 1

    img_pro = np.copy(img_bin)

    # apply circular mask
    img_pro[(plate_mask == False)] = 255 * inv
    img_pro = crop_image(img_pro, circle)

    if inv == 0:
        img_show = img_pro
    elif inv == 1:
        img_show = cv2.bitwise_not(img_pro)

    if DEBUG:
        output_image("5. preprocessed", img_show)

    return img_ori, img_show


def find_plate(img_ori, img_bin):
    # define the max possible plate radius as
    max_possible_radius = int(min(img_bin.shape) / 2)
    circle = 0
    radius_scale = 35 / 100
    max_radius = int((max_possible_radius * radius_scale) + (max_possible_radius * 0.5))
    min_radius = max_radius - 10
    radius_offset = 100 / 100

    # find plate in the image with Hough Circle Transform
    circles = cv2.HoughCircles(img_bin, cv2.HOUGH_GRADIENT, 1, 20, param1=100,
                                    param2=10, minRadius=min_radius, maxRadius=max_radius)

    img_show = img_ori.copy()

    if circles is not None:
        # return data of the smallest circle found
        circles = (circles[0, :]).astype("float")
        max_c = np.argmax(circles, axis=0)
        indx = max_c[2]
        circle = circles[indx]
        circle = (int(circle[0]), int(circle[1]), int(radius_offset * circle[2]))
        # draw the outer circle
        cv2.circle(img_show, (circle[0], circle[1]), circle[2], houghColor, 2)
        # draw the center of the circle
        cv2.circle(img_show, (circle[0], circle[1]), 2, houghColor, 3)

    plate_mask = np.zeros(img_bin.shape, np.uint8)
    plate_mask = cv2.circle(plate_mask, (circle[0], circle[1]), circle[2], (255, 255, 255),
                            thickness=-1)

    return plate_mask, circle

def hough_circle_method(img_ori, img_pro):
    min_radius = 15
    max_radius = 15

    # find circles is the image with Hough Circle Transform
    circles = cv2.HoughCircles(img_pro, cv2.HOUGH_GRADIENT, sensitivity+1, nhood+1, param1=100,
                                    param2=param2+1, minRadius=min_radius+1, maxRadius=max_radius+1)

    img_show = img_ori.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(img_show, (i[0], i[1]), i[2], houghColor, 2)

    if DEBUG:
        output_image("6. hough_circle", img_show)

    return img_show, len(circles[0])

def load_image(image, desired_res):
    img_ori = image
    scale = (desired_res / img_ori.shape[1])
    img_ori = cv2.resize(img_ori, (int(img_ori.shape[1] * scale), int(img_ori.shape[0] * scale)))
    return img_ori

def crop_image(img, mask):
    output = img

    # if the height is greater than the width (portrait)
    if img.shape[0] > img.shape[1]:
        x_pos, y_pos, radius = mask
        x_bot = int(x_pos - radius)    
        y_bot = int(y_pos - radius)    
        x_top = int(x_pos + radius)    
        y_top = int(y_pos + radius)    

        # find min distance from the edge of the box to the image border
        min_x_dist = min((img.shape[1] - x_top), (img.shape[1] - (img.shape[1] - x_bot)))
        min_y_dist = min((img.shape[0] - y_top), (img.shape[0] - (img.shape[0] - y_bot)))
        min_dist = min(min_x_dist, min_y_dist)

        x_bot = (x_bot - min_dist)    
        y_bot = (y_bot - min_dist)    
        x_top = (x_top + min_dist)    
        y_top = (y_top + min_dist)    

        # crop image using the new mask
        output = output[y_bot:y_top, x_bot:x_top]

    return output

def generate_output(img, count, filename):
    name = filename.split("/")[-1].split(".")[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scl = 0.5
    font_col = (0, 0, 0)

    # create border
    height, width, chan = img.shape
    border_size = 20
    image_height = height + (border_size + 60)
    image_width = width + border_size
    border = np.zeros((image_height, image_width, chan), np.uint8)
    border[:,0:image_width] = houghColor

    file_str = 'File: "{}"'.format(name)
    cv2.putText(border, file_str, (10, image_height-40), font, font_scl, font_col, 1, cv2.LINE_AA)

    count_str = 'Vibrios: {}'.format(count)
    cv2.putText(border, count_str, (10, image_height-20), font, font_scl, font_col, 1, cv2.LINE_AA)

    border[10:10+height, 10:10+width] = img
    output_image('7. {}_V{}'.format(name, count), border)
    
    return border

uploaded_file = st.file_uploader("Upload the image")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    img_ori = load_image(img_array, 980)
    img_ori, img_pro = pre_process(img_ori)
    output, vibrios = hough_circle_method(img_ori, img_pro)
    output = generate_output(output, vibrios, uploaded_file.name)
    
    # write the result
    st.image(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
    st.write("Vibrio count: "+str(vibrios))