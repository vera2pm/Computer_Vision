"""
Methods to perm=form last stage of pan-sharpening -- combine panchromatic image with colored image
"""
import cv2


def brovey_transform(img, gray_img):
    b, g, r = cv2.split(img)
    const = gray_img * 3 / (b + g + r)
    r_out = r * const
    b_out = b * const
    g_out = g * const
    return cv2.merge((b_out, g_out, r_out))


def intensity_transform(img, gray_img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, satur, value = cv2.split(hsv)
    sharpen = cv2.merge((hue, satur, gray_img))
    sharpen = cv2.cvtColor(sharpen, cv2.COLOR_HSV2RGB)
    return sharpen
