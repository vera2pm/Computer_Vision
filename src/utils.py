import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1


def find_keypoints(img):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    keypoints, des = sift.detectAndCompute(img, None)
    return keypoints, des


def draw_matches(gray_img, keypoints1, rgb_half, keypoints2, good, matchesMask):
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    imag_matches = cv2.drawMatches(gray_img, keypoints1, rgb_half, keypoints2, good, None, **draw_params)
    plt.figure(figsize=(15, 7))
    plt.imshow(imag_matches, 'gray')
    plt.show()


def find_homo(gray_img, rgb_half):

    keypoints1, des1 = find_keypoints(gray_img)
    keypoints2, des2 = find_keypoints(rgb_half)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        # h, w, d = gray_img.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, matrix)
        # rgb_half_lines = cv2.polylines(rgb_half, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_matches(gray_img, keypoints1, rgb_half, keypoints2, good, matchesMask)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        matrix = None

    return matrix

