"""
Methods to perform last stage of pan-sharpening -- combine panchromatic image with colored image
"""
import cv2
import numpy as np


def brovey_transform(img, gray_img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img)
    const = gray_img / ((r + g + b)/3)
    r_out = const * r
    b_out = const * b
    g_out = const * g
    new_img = np.float32(cv2.merge((b_out, g_out, r_out)))
    print(new_img.shape)
    return cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)


def intensity_transform(img, gray_img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, satur, value = cv2.split(hsv)
    sharpen = cv2.merge((hue, satur, gray_img))
    sharpen = cv2.cvtColor(sharpen, cv2.COLOR_HSV2RGB)
    return sharpen
