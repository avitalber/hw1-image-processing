from hw1_functions import *
import cv2
import numpy as np

if __name__ == "__main__":

    path_image = r'Images\darkimage.tif'
    darkimg = cv2.imread(path_image)
    darkimg_gray = cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY)

    print("Start running script  ------------------------------------\n")
    print_IDs()

    print("a ------------------------------------\n")
    enhanced_img, a, b = contrastEnhance(darkimg, [0,255])#add parameters

    # display images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title('original')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)
    plt.title('enhanced contrast')

    # print a,b
    print("a = {}, b = {}\n".format(a, b))

    minval= np.min(darkimg_gray)
    maxval= np.max(darkimg_gray)
    print("MAX:" , maxval, "MIN:", minval)

    #display mapping
    showMapping([minval, maxval], a, b)

    print("b ------------------------------------\n")
    enhanced2_img, a, b = contrastEnhance(enhanced_img, [0,255])
    # print a,b
    print("enhancing an already enhanced image\n")
    print("a = {}, b = {}\n".format(a, b))

    # TODO: display the difference between the two image (Do not simply display both images)
    dist_between_enhanced = minkowski2Dist(enhanced2_img, enhanced_img)  # add parameters

    print("Minkowski dist between enhanced and enhanced2\n")
    print("d = {}\n".format(dist_between_enhanced))



    print("c ------------------------------------\n")
    mdist = minkowski2Dist(enhanced_img, enhanced_img)
    print("Minkowski dist between image and itself\n")
    print("d = {}\n".format(mdist))



    # implementation of the loop that calculates minkowski distance as function of increasing contrast:

    contrast = []
    my_range= []
    dists = []
    num_of_iterations= 20
    lower_bound=minval
    top_bound=0
    i=1
    k=((maxval-minval)/num_of_iterations)


    while num_of_iterations:
        top_bound = round(minval + i * k)
        my_range.insert(0,lower_bound)
        my_range.insert(1,top_bound)
        nim, a, b = contrastEnhance(darkimg_gray, my_range)
        dist= minkowski2Dist(darkimg_gray, nim)
        dists.insert(i-1,dist)
        contrast.append(top_bound-minval)
        top_bound+=k
        i+=1
        my_range.clear()
        num_of_iterations-=1


    plt.figure()
    plt.plot(contrast, dists)
    plt.xlabel("contrast")
    plt.ylabel("distance")
    plt.title("Minkowski distance as function of contrast")

    print("d ------------------------------------\n")

    #computationally proof that sliceMat(im) * [0:255] == im
    enhanced_img_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    a = np.arange(256)
    x = mapImage(enhanced_img_gray, a)
    d = meanSqrDist(x, enhanced_img_gray)
    print("d={}".format(d))


    print("e ------------------------------------\n")
    # working on enhanced image gray and original image

    # TMim, TM = SLTmap(darkimg_gray, enhanced_img_gray)
    # vec = np.array([range(0, 256)])
    # d = meanSqrDist(darkimg_gray, np.matmul(sliceMat(darkimg_gray), np.transpose(vec)).reshape(256, 256))  # computationally compare

    TMim, SLTim  = SLTmap(darkimg_gray, enhanced_img_gray)
    toned_im = np.matmul(sliceMat(darkimg_gray), SLTim).reshape(enhanced_img_gray.shape)
    # calculate with meanSqrDist
    d = meanSqrDist(toned_im, TMim)

    print("sum of diff between image and slices*[0..255] = {}".format(d))

    # then display
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title("original image")
    plt.subplot(1, 2, 2)
    plt.imshow(TMim, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")



    print("f ------------------------------------\n")
    negative_im = sltNegative(darkimg_gray)
    plt.figure()
    plt.imshow(negative_im, cmap='gray', vmin=0, vmax=255)
    plt.title("negative image using SLT")




    print("g ------------------------------------\n")
    thresh = 120 # play with it to see changes
    lena = cv2.imread(r"Images\\RealLena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    thresh_im = sltThreshold(lena_gray, thresh)

    plt.figure()
    plt.imshow(thresh_im, cmap='gray', vmin=0, vmax=255)
    plt.title("thresh image using SLT")


    print("h ------------------------------------\n")

    im1 = lena_gray
    im2 = darkimg_gray

    SLTim, tm = SLTmap(im1, im2)

    # then print
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im1,cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(SLTim, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")
    plt.subplot(1, 3, 3)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")


    d1 = meanSqrDist(im1, im2)
    d2 = meanSqrDist(SLTim, im2)
    print("mean sqr dist between im1 and im2 = {}\n".format(d1))
    print("mean sqr dist between mapped image and im2 = {}\n".format(d2))



    print("i ------------------------------------\n")
    SLTim2, _= SLTmap(im2,im1)
    d= minkowski2Dist(SLTim, SLTim2)
    print(" {}".format(d))

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(SLTim, cmap='gray', vmin=0, vmax=255)
    plt.title("SLTmap(im1,im2)")

    plt.subplot(1, 4, 2)
    plt.imshow(SLTim2, cmap='gray', vmin=0, vmax=255)
    plt.title("SLTmap(im2,im1)")

    plt.show()