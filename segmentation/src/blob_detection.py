import cv2
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
import matplotlib.pyplot as plt


def blob_detection(img, i):
    if type(img) == str:
        hotdog = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    else:
        hotdog = img
        hotdog = cv2.cvtColor(hotdog, cv2.COLOR_BGR2GRAY)
    # hotdog = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # hotdog = hotdog.astype(np.uint8)
    # print(f"max {np.max(hotdog)}")

    # hotdog = cv2.cvtColor(hotdog, cv2.COLOR_BGR2GRAY)
    hotdog = cv2.normalize(hotdog, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # hotdog = hotdog.astype(np.uint8)
    print(hotdog.shape)

    # blobs = blob_log(hotdog, max_sigma=30, num_sigma=10, threshold=60)
    blobs = blob_dog(hotdog, max_sigma=100, threshold=60)
    print(len(blobs))

    fig, ax = plt.subplots()
    ax.imshow(hotdog, cmap="gray")
    for blob in blobs:
        y, x, area = blob
        ax.add_patch(plt.Circle((x, y), area * np.sqrt(2), color="r", fill=False))
    # plt.show()
    fig.savefig(f"../data/test_ima/test_mask_{i}.jpg")
    return len(blobs)


def blob_detection_1(img, i):
    """
    Function to detect blob in one image
    :param img:
    :return:
    """
    # Read image
    if type(img) == str:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    img = 255 - img
    # print(img)

    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(img)
    print(len(keypoints))

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(img, keypoints, blank, (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    fig, ax = plt.subplots()
    ax.imshow(blobs)
    # plt.show()
    # fig = plt.figure(figsize=(15, 5))
    # plt.imshow(blobs)
    # fig.suptitle("title")
    cv2.imwrite(f"../key_points.jpg", blobs)
    fig.savefig(f"../data/test_ima/test_mask_{i}.jpg")
    # plt.savefig(f'../key_points.jpg')
    print("Saved file")
    return len(keypoints)
