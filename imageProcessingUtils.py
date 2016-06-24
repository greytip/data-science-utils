from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters
from skimage.color import rgb2gray

@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)

@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)

@adapt_rgb(as_gray)
def sobel_gray(image):
    return filters.sobel(image)

def image_as_gray(image_filter, image, *args, **kwargs):
    gray_image = rgb2gray(image)
    return image_filter(gray_image, *args, **kwargs)

def makeBorder(img, btype=None, **kwargs):
    import cv2
    if not btype:
        btype=cv2.BORDER_REPLICATE
    cv2.copyMakeBorder(img1,10,10,10,10,btype,**kwargs)

def blendImages(img1, img2):
    import cv2
    return cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

def bitwiseOps(img1, img2=None, ops=None):
    assert ops, "Please pass a bit operation"
    if ops='not':
        return cv2.bitwise_not(img)
    if ops='add':
        assert img1 and img2
        return cv2.bitwise_and(img1, img2)

def translateImage(img, dist=(0,0)):
    import cv2
    import numpy as np
    rows, cols = img.shape
    M = np.float32([[1, 0, dist[0]], [0,1, dist[1]]])
    return cv2.warpAffine(img, M, (cols, rows))

def rotateImage(img, degree):
    import cv2
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), degree, 1)
    return cv2.warpAffine(img, M, (cols, rows))

def perspectiveTransform(img, orig_pt, pers_pt):
    import cv2
    rows, cols, ch = img.shape
    # TODO: add a co-linear check..
    assert(len(orig_pt) ==4), "Please pass 4 points in original image"
    assert(len(pers_pt) ==4), "Please pass 4 points in perspective image"
    pts1 = np.float32(orig_pt)
    pts2 = np.float32(pers_pt)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,(300,300))

