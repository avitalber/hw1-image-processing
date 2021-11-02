import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def print_IDs():
    print("206181554+205815111\n")


def contrastEnhance(im, range1):
    # TODO: implement fucntion
    # height, width, number of channels in image
    height = im.shape[0]
    width = im.shape[1]
    minval=np.min(im)
    maxval= np.max(im)
    # calculating the ols contrast
    c_old = maxval-minval
    # calculating the new contrast
    c_new = range1[1]-range1[0]
    #finding a
    a = c_new/c_old
    #finding b by placing a
    b = range1[1]-a*maxval

    nim=np.copy(im)

    #changing the image
    for x in range(height):
        for y in range(width):
            nim[x, y] = a*im[x, y]+b

    return nim, a, b


def showMapping(old_range, a, b):
    imMin = np.min(old_range)
    imMax = np.max(old_range)
    x = np.arange(imMin, imMax+1, dtype=np.float)
    y = a * x + b
    plt.figure()
    plt.plot(x, y)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.title('contrast enhance mapping')
    plt.show()


def minkowski2Dist(im1,im2):
    #creating 2 normalized histogram
    hist1=np.histogram(im1, bins=256, range=(0,255))
    hist2=np.histogram(im2, bins=256, range=(0,255))
    hist1_im=hist1[0].astype(np.float)/((hist1[0].shape[0])*(hist1[1].shape[0]))
    hist2_im=hist2[0].astype(np.float)/((hist2[0].shape[0])*(hist2[1].shape[0]))
    temp=0.0

    #implementing the Minkowski formula
    for i in range(0,256):
        temp+=np.power(hist1_im[i]-hist2_im[i],2)

    d= np.power(temp, 0.5)
    return d


def meanSqrDist(im1, im2):
    d=np.mean(np.power(im1.astype(float)-im2.astype(float), 2)).astype(float)
    return d


def sliceMat(im):
    im_height = im.shape[0]
    im_weight = im.shape[1]
    row_matrix = im_height * im_weight
    Slices = np.zeros((row_matrix, 256))
    i = 0
    for index, val in np.ndenumerate(im):
        Slices[i, val] = 1
        i+=1

    return Slices


def SLTmap(im1, im2):
    im_slice = sliceMat(im1)
    TM = np.empty((256), int)
    im2_flatt = np.ravel(im2)
    for g in range(0,256):
        col = np.sum(im_slice[:,g])
        if col == 0 :
            TM[g] = 0

        else:
            n_mat = im_slice[:,g]*im2_flatt
            mean = (np.sum(n_mat) / col).astype(float)
            TM[g] = mean
    return mapImage(im1, TM), TM


def mapImage (im,tm):
    im_slice = sliceMat(im)
    imVec = np.matmul(im_slice, tm)
    TMim = np.reshape(imVec, (len(im), -1))
    return TMim


def sltNegative(im):
    nim= mapImage(im, np.flip(np.arange(256)))
    return nim


def sltThreshold(im, thresh):
    array = np.arange(256)
    #changing the array by threshold:
    for i in range(256):
        if i<=thresh:
            array[i]=0
        else:
            array[i]=255
    #creating the new image:
    nim= mapImage(im, array)
    return nim