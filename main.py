# Import TF and TF Hub libraries.
import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
from util.demo import *
from util.model import *
def show_res(img, results, path_name = "out.png"):
    for _, item in enumerate(results):
        cv.circle(img, (int(item[1]), int(item[0])), 5, (0, 255, 0), 3)
    cv.imwrite(path_name, img)
# Load the input image.
def pre_process(image_path):
    if isinstance(image_path, str):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)
    else:
        image = image_path
    input_size = 256
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    return input_image, image
image_path = './imgs/input_image.jpeg'

video_path = "./imgs/pose.mp4"
cap = cv.VideoCapture(video_path)
while True:
    ret, img = cap.read()
    if ret == None:
        break
    img = tf.convert_to_tensor(img)
    input_image, ori = pre_process(img)
    keypoints_with_scores = movenet(input_image)
    
    # Visualize the predictions with image.
    display_image = tf.expand_dims(ori, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
        display_image, 640, 640), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

    # plt.figure(figsize=(5, 5))
    # plt.imshow(output_overlay)
    cv.imshow('h', cv.cvtColor(output_overlay, cv.COLOR_BGR2RGB))
    if cv.waitKey(10) == ord("q"):
        cv.destroyAllWindows()
