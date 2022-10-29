import cv2
import math as m
import numpy as np
import random
import time

start = time.time()

img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')


def ORB(): #finds corresponding points
    orb = cv2.ORB_create(nfeatures=1800)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    src_keypoints = []
    dst_keypoints = []
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        src_keypoints.append((y1, x1))
        dst_keypoints.append((y2, x2))
    return np.array(src_keypoints), np.array(dst_keypoints)


def DLT(srcX, dstXprime):
    total_x, total_y, total_x_prime, total_y_prime = 0, 0, 0, 0
    for i in range(0, len(srcX)):
        total_x += srcX[i][0]
        total_y += srcX[i][1]
        total_x_prime += dstXprime[i][0]
        total_y_prime += dstXprime[i][1]

    avg_ref_x = total_x / len(srcX)
    avg_ref_y = total_y / len(srcX)
    dst_avg_x = total_x_prime / len(dstXprime)
    dst_avg_y = total_y_prime / len(dstXprime)

    total_distanceX, total_distanceXprime = 0, 0
    for i in range(0, len(srcX)):  # MOVING OPERATION
        dist = m.sqrt( ((srcX[i][0] - avg_ref_x) ** 2) + ((srcX[i][1] - avg_ref_y)** 2))
        total_distanceX += dist
        dist = m.sqrt( ((dstXprime[i][0] - dst_avg_x) ** 2) + ((dstXprime[i][1] - dst_avg_y) ** 2))
        total_distanceXprime += dist

    scaleX = m.sqrt(2) / (total_distanceX / len(srcX))
    scaleXprime = m.sqrt(2) / (total_distanceXprime / len(dstXprime))

    normalizedX = []  # normalized ref points
    normalizedXprime = []  # normalized transformated points
    for i in range(0, len(srcX)):
        normalizedX.append(((srcX[i][0] - avg_ref_x) * scaleX, (srcX[i][1] - avg_ref_y) * scaleX))
        normalizedXprime.append(((dstXprime[i][0] - dst_avg_x) * scaleXprime, (dstXprime[i][1] - dst_avg_y) * scaleXprime))

    normalizedX = np.array(normalizedX)
    normalizedXprime = np.array(normalizedXprime)

    T = np.array([[scaleX, 0, -avg_ref_x * scaleX],
                  [0, scaleX, -avg_ref_y * scaleX],
                  [0, 0, 1]])

    Tprime = np.array([[scaleXprime, 0, -dst_avg_x * scaleXprime],
                       [0, scaleXprime, -dst_avg_y * scaleXprime],
                       [0, 0, 1]])

    A = []
    for i in range(0, len(srcX)):  # CONSTRUCT MATRIX A
        x = normalizedX[i][0]
        y = normalizedX[i][1]
        xprime = normalizedXprime[i][0]
        yprime = normalizedXprime[i][1]

        A.append([0, 0, 0, -x, -y, -1, yprime * x, yprime * y, yprime])
        A.append([x, y, 1, 0, 0, 0, -xprime * x, -xprime * y, -xprime])

    A = np.array(A)
    U, S, Vh = cv2.SVDecomp(np.asarray(A))
    L = Vh[-1]
    H = L.reshape(3, 3)

    invTprime = np.linalg.inv(Tprime)
    H1 = np.dot(invTprime, H)
    finalHomography = np.dot(H1, T)

    return finalHomography


def applyHomography(H, srcX): #takes X as parameter and find X' = HX
    XprimeTemp = []
    for k in range(0, len(srcX)):
        Xtemp = np.array([[srcX[k][0]], [srcX[k][1]], [1]])
        newXprimetemp = np.dot(H, Xtemp)  # POINTS ARE TRANSFORMED

        XprimeXcoor = newXprimetemp[0] / newXprimetemp[2]
        XprimeYcoor = newXprimetemp[1] / newXprimetemp[2]

        XprimeTemp.append((XprimeXcoor[0], XprimeYcoor[0]))

    XprimeTemp = np.array(XprimeTemp)
    return XprimeTemp


def RANSAC():
    maxInlierCount = 0
    maxHomography = []

    N = 10000
    p = 0
    while (p < N):  # RANSAC
        numberOfInliers = 0
        inlierX = []
        inlierXprime = []

        X1 = []
        X1prime = []
        for j in range(0, 4):
            ran = random.randint(0, len(src_points) - 1)
            X1.append(src_points[ran])
            X1prime.append(dst_points[ran])

        X1 = np.array(X1)
        X1prime = np.array(X1prime)

        homography = DLT(X1, X1prime)

        transformedX = applyHomography(homography, src_points)

        for t in range(0, len(transformedX)):
            for g in range(0, len(dst_points)):
                p1x = dst_points[g][0] - transformedX[t][0]
                p1y = dst_points[g][1] - transformedX[t][1]
                distance = int(m.sqrt((p1x ** 2) + (p1y ** 2)))

                if (distance <= 3):
                    inlierX.append([src_points[g][0], src_points[g][1]])
                    inlierXprime.append([dst_points[g][0], dst_points[g][1]])
                    numberOfInliers += 1
                    break

        print("Loop = ", p, "Inlier count: ", numberOfInliers, " N: ", N)

        w1 = (numberOfInliers / len(src_points))
        ws = w1 ** 4
        denominator = np.log(1.0 - ws)

        newN = 0
        if (denominator != 0.0):

            newN = int(abs(m.log(0.01) / denominator))

            if (newN > N):
                p += 1
                continue
            else:
                N = newN
                maxHomography = homography
                maxInlierCount = numberOfInliers
                maxInlierX = inlierX
                maxInlierXprime = inlierXprime

        print("Loop = ", p, " Calculated New N :", newN, "Continue with N = ", N)
        p += 1

    print("Max Homography after RANSAC:\n", maxHomography)
    print("Max inlier after RANSAC :> ", maxInlierCount)
    return np.array(maxInlierX), np.array(maxInlierXprime), maxHomography, maxInlierCount


# -------------------------------
# *********MAIN******************
src_points, dst_points = ORB()
print("There are ", len(src_points), " matched keypoint!")

maxInlierX, maxInlierXprime, maxHomography, maxInlierCount = RANSAC()

homog = DLT(maxInlierX, maxInlierXprime)
transformedX = applyHomography(homog, maxInlierX)
print("New estimated Homography:\n", homog)

# STEP 6-7-8 IN HOMEWORK
while (True):
    inlierCount = 0

    inlierXtemp = []
    inlierXprimeTemp = []
    for t in range(0, len(transformedX)):
        for g in range(0, len(maxInlierXprime)):
            p1x = maxInlierXprime[g][0] - transformedX[t][0]
            p1y = maxInlierXprime[g][1] - transformedX[t][1]
            distance = m.sqrt((p1x ** 2) + (p1y ** 2))

            if (distance <= 3):
                inlierXtemp.append([maxInlierX[g][0], maxInlierX[g][1]])
                inlierXprimeTemp.append([maxInlierXprime[g][0], maxInlierXprime[g][1]])
                inlierCount += 1
                break

    print("Inlier count: ", inlierCount)

    if (inlierCount < 100):
        homog = maxHomography
        break
    if (inlierCount <= maxInlierCount):
        if (maxInlierCount == inlierCount):
            print("CONVERGED")
            break

        maxInlierCount = inlierCount

    homog = DLT(inlierXtemp, inlierXprimeTemp) #every step update homography and inliers
    transformedX = applyHomography(homog, inlierXtemp)

print("Final homography:\n", homog)
print("Operation takes:", (time.time() - start), "s")

x1 = np.dot(homog, [160, 160, 1])
x2 = np.dot(homog, [160, 464, 1])
x3 = np.dot(homog, [575, 171, 1])
x4 = np.dot(homog, [575, 450, 1])

a1x = int(x3[1] / x3[2])  # leftmost corner
a1y = int(x3[0] / x3[2])
b1x = int(x2[1] / x2[2])  # rightbottom corner
b1y = int(x2[0] / x2[2])

img3 = cv2.imread('img2a.jpg')
img3 = cv2.rectangle(img3, (a1x, a1y), (b1x, b1y), (0, 0, 255), 2)  # if points are valid, draw rectangle

cv2.imwrite("final_image.png", img3)

cv2.imshow('Display Image', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''COMMENT 1  from(Image Matching Using SIFT, SURF, BRIEF and ORB: Performance Comparison for Distorted Images Ebrahim Karami, Siva Prasad, and Mohamed Shehata
                                                                                            Faculty of Engineering and Applied Sciences, Memorial University, Canada

I have readed details of these detectors in that paper and my summation is at below: 
        
SIFT is a feature detector which requires high computational complexity.It has 4 steps
 (1) Scale Space Extrema Detection, (2) Key point Localization, (3) Orientation Assignment and (4) Description Generation

ORB is a combination of the FAST key point detector and BRIEF descriptor with some modifications.
Determination of keypoints is done by FAST and ORB overcomes the lack of rotation invariance of BRIEF.

Comparison about recognition performance under image deformations:

Computation time in ORB is less than SIFT because of computational complexity of SIFT.
SIFT has better matching rate under varying intensity, rotation, but ORB is better under the scaling.
As a result SIFT is better in performance under almost all image deformations according to tables of experiments in that paper,
but ORB is faster than SIFT.Also that conclusion sentence was critical in that paper:
In ORB, the features are mostly concentrated in objects at the center of the image while in SIFT  key point detectors are
distributed over the image.


'''

'''  COMMENT 2
I find nearest neighbour using cv2.BFMatcher() function of OpenCV in ORB.It takes normType as parameter and 
it specifies the distance measurement to be used.
In ORB I used cv2.NORM_HAMMING ,which used Hamming distance as measurement. 
In SIFT I used cv2.NORM_L2.

'''




'''  COMMENT 3

In RANSAC, our aim is to find inlier set of matches after transformation of source points with H which is 
calculated from DLT with 4 random correspondences.Choosing appropriate treshold is important to eliminate
outliers.If we select treshold smaller than 3, we can eliminate more outliers and we can have more robust
homography at the end, but choosing very small treshold may cause eliminate inliers which are good enough
to have robust homography at the end and this is also not good.

'''


'''  COMMENT 4

In DLT, we have some source points in an image and we also know the correspondences of these points.
In my opinion, the aim of normalizing all these source and destination points is to eliminate huge distance differences.
Dividing square 2 by average distance of source and destination points to center(0,0) makes changes minimum and
that decreases the error rate.

'''
