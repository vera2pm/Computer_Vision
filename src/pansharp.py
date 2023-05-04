"""
Methods to perform last stage of pan-sharpening -- combine panchromatic image with colored image
"""
import cv2
import numpy as np


def brovey_transform(img, gray_img):
    print(gray_img[250, 250])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img)
    print(b[250, 250])
    print(g[250, 250])
    print(r[250, 250])
    # const = gray_img / (1/3*(r + g + b))
    # print(const[250, 250])
    ml = (r + g + b) / 3
    print(type(ml))
    print(gray_img.shape)
    print(ml.shape)
    print(r.shape)
    r_out = gray_img * r / ml
    b_out = gray_img * b / ml
    g_out = gray_img * g / ml

    print(b_out[250, 250])
    print(g_out[250, 250])
    print(r_out[250, 250])
    new_img = np.float32(cv2.merge((b_out, g_out, r_out)))
    print(new_img.shape)
    return cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)


def intensity_transform(img, gray_img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, satur, value = cv2.split(hsv)
    sharpen = cv2.merge((hue, satur, gray_img))
    sharpen = cv2.cvtColor(sharpen, cv2.COLOR_HSV2RGB)
    return sharpen
