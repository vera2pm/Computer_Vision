"""
Methods to perform last stage of pan-sharpening -- combine panchromatic image with colored image
"""
import cv2
import numpy as np


def brovey_transform(img, gray_img):
    print(gray_img.dtype)
    print(img.dtype)
    print(gray_img.shape)
    print(img.shape)
    if gray_img.dtype == np.uint8:
        gray_img = gray_img.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img)
    ml = (r + g + b) / 3
    r_out = gray_img * r / ml
    b_out = gray_img * b / ml
    g_out = gray_img * g / ml

    new_img = np.float32(cv2.merge((b_out, g_out, r_out)))
    if new_img.dtype == np.float32:
        new_img = (new_img * 255.0).astype(np.uint8)
    print(new_img.shape)
    return cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)


def intensity_transform(img, gray_img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, satur, value = cv2.split(hsv)
    sharpen = cv2.merge((hue, satur, gray_img))
    sharpen = cv2.cvtColor(sharpen, cv2.COLOR_HSV2RGB)
    return sharpen
