import tensorflow as tf
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

def state_placeholder():
    return tf.placeholder([84,84])

def preprocess_image_tf(image):
    new_image = tf.image.rgb_to_grayscale(image)
    new_image = tf.image.resize_image_with_crop_or_pad(new_image, 160, 160)
    new_size = tf.constant([84,84])
    new_image = tf.image.resize_images(new_image,new_size)
    return new_image

def crop_center(image, tsize):
    dx,dy = image.shape[:2]
    adjust_dx = max(0, dx-tsize[0])
    adjust_dy = max(0, dy-tsize[1])
    return image[ adjust_dx // 2 : tsize[0] - 1 - (adjust_dx // 2),
                   adjust_dy // 2 : tsize[1] - 1 - (adjust_dy // 2)]

def preprocess_image(image):
    grays = rgb2gray(image)
    grays = crop_center(grays, (160, 160))
    resized = resize(grays, (84, 84))
    return resized

def l2_loss(estimate,target):
    l1_loss=tf.abs(estimate-target)
    return l1_loss*l1_loss
