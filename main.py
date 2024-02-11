import math

import cv2
import numpy as np
import imutils

image_path1 = 'apple.jpg'
image_path2 = 'orange.jpg'

y_offset = 32
y_offset_copy = 1

original_image1 = cv2.imread(image_path1)
original_image2 = cv2.imread(image_path2)
kernel = np.array([[0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625],
       [0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625  ],
       [0.0234375 , 0.09375   , 0.140625  , 0.09375   , 0.0234375 ],
       [0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625  ],
       [0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625]])

def find_max_pyramid_level():
    min_dimension = min(original_image1.shape[0], original_image1.shape[1])
    max_pyramid_level = int(np.floor(np.log2(min_dimension))) - 1
    return max_pyramid_level

def downsample(image):
    image = cv2.filter2D(image,-1,kernel)
    return image[::2, ::2]

def upsample(image):
    image1_temp = np.zeros((image.shape[0]*2,image.shape[1]*2,3))
    image1_temp[::2, ::2] = image
    image1_temp = cv2.filter2D(image1_temp,-1,kernel*4)
    return image1_temp

def build_gaussian_pyramid(image):
    image_copy = image.copy()
    gaussian = [image_copy]
    for i in range(pyramid_levels):
        image_copy=downsample(image_copy)
        gaussian.append(image_copy)
    return gaussian

def build_Laplacian_From_Gaussian(image):
    gaussian = build_gaussian_pyramid(image)
    image_copy = gaussian[pyramid_levels-1]
    laplacian = [image_copy]
    for i in range (pyramid_levels-1,0,-1):
        gaussian_expand = upsample(gaussian[i])
        try:
            temp = cv2.subtract(gaussian[i-1],gaussian_expand, dtype=cv2.CV_32F)
        except:
            size = tuple(reversed(np.shape(gaussian_expand)[:2]))
            gaussian[i-1]=cv2.resize(gaussian[i-1],size)
            temp = cv2.subtract(gaussian[i - 1], gaussian_expand, dtype=cv2.CV_32F)
        laplacian.append(temp)
    return laplacian

def create_RegionMaskPyramid_From_Image(image):
    region_mask = np.zeros(np.shape(image))
    region_mask2 = np.ones(np.shape(image))
    roi = cv2.selectROI(image)
    region_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1


    if(checkImagesEqual()):
        roi = (roi[0], roi[1] + y_offset, roi[2], roi[3])

    region_mask2[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 0

    region_mask_pyramid = build_gaussian_pyramid(region_mask)
    region_mask_pyramid2 = build_gaussian_pyramid(region_mask2)

    return region_mask_pyramid,region_mask_pyramid2

def checkImagesEqual():
    difference = cv2.subtract(original_image1, original_image2)
    b, g, r = cv2.split(difference)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        return True
    return False


pyramid_levels = find_max_pyramid_level()

if(checkImagesEqual()):
    pyramid_levels = int(math.log2(abs(y_offset))) + 1

laplacian1 = build_Laplacian_From_Gaussian(original_image1)
laplacian2 = build_Laplacian_From_Gaussian(original_image2)
gaussianMask1, gaussianMask2 = create_RegionMaskPyramid_From_Image(original_image1)

blended_images = []
for i in range(pyramid_levels):
    try:
        temp1 = cv2.multiply(laplacian1[i]/255,gaussianMask1[pyramid_levels-(i+1)], dtype=cv2.CV_32F)
    except:
        size = tuple(reversed(np.shape(laplacian1[i]/255)[:2]))
        gaussianMask1[pyramid_levels-(i+1)] = cv2.resize(gaussianMask1[pyramid_levels-(i+1)], size)
        temp1 = cv2.multiply(laplacian1[i]/255,gaussianMask1[pyramid_levels-(i+1)], dtype=cv2.CV_32F)

    if(checkImagesEqual()):
        temp1 = imutils.translate(temp1,0,y_offset_copy)

    try:
        temp2 = cv2.multiply(laplacian2[i]/255, gaussianMask2[pyramid_levels-(i+1)], dtype=cv2.CV_32F)
    except:
        size = tuple(reversed(np.shape(laplacian2[i]/255)[:2]))
        gaussianMask2[pyramid_levels-(i+1)] = cv2.resize(gaussianMask2[pyramid_levels-(i+1)], size)
        temp2 = cv2.multiply(laplacian2[i]/255, gaussianMask2[pyramid_levels-(i+1)], dtype=cv2.CV_32F)

    blended_image = cv2.add(temp1,temp2)
    blended_images.append(blended_image)
    y_offset_copy=y_offset_copy*2

blended_reconstruct = blended_images[0]

for i in range (1,pyramid_levels):
    blended_reconstruct = upsample(blended_reconstruct)
    blended_reconstruct = cv2.add(blended_images[i], blended_reconstruct, dtype=cv2.CV_32F)
    resized_image = cv2.resize(blended_reconstruct, tuple(reversed(np.shape(original_image2)[:2])), interpolation=cv2.INTER_LINEAR)
cv2.imshow("last" , resized_image)

cv2.waitKey(0)