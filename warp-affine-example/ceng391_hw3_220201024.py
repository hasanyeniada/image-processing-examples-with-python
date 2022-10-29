import numpy as np
import math as k
import cv2

img = cv2.imread('img1.png', 0)

#2 x 2 affine matrix
def create_affine(scale, rot_angle, tilt):
    r = -1 * rot_angle

    a1 = np.array([[k.cos(k.radians(rot_angle)), k.sin(k.radians(rot_angle))],
                   [k.sin(k.radians(r)), k.cos(k.radians(rot_angle))]])

    a2 = np.array([[1/k.cos(k.radians(tilt)), 0],
                   [0, 1]])

    A = np.dot(a1, a2)
    A = A * scale

    return A


A = create_affine(0.5, 30, 50)
print("Affine 2x2:\n ", A ,"\n")

c1 = np.asarray( [ 0,0 ])
c2 = np.asarray( [ 0, img.shape[1] ] )
c3 = np.asarray( [ img.shape[0], 0 ]  )
c4 = np.asarray( [ img.shape[0], img.shape[1] ]  )

xs = []
ys = []

w1 = np.dot(A, c1) #correspondence of corner(0,0)
x1, y1 = w1[0], w1[1]
xs.append(x1)
ys.append(y1)

w2 = np.dot(A, c2) #correspondence of corner(800,0)
x1, y1 = w2[0], w2[1]
xs.append(x1)
ys.append(y1)

w3 = np.dot(A, c3) #correspondence of corner(0,640)
x1, y1 = w3[0], w3[1]
xs.append(x1)
ys.append(y1)

w4 = np.dot(A, c4) #correspondence of corner(640,800)
x1, y1 = w4[0], w4[1]
xs.append(x1)
ys.append(y1)


minx = min(xs)
maxx = max(xs)
miny = min(ys)
maxy = max(ys)

wH = int(maxx-minx) #output image height
wW = int(maxy-miny) #output image width

rW = img.shape[1] #reference image width
rH = img.shape[0] #reference image height

print("Warped Image Height:", wH, "Warped Image Width:", wW, "\n" )

# 3 x 3 AFFINE
def create_3x3_affine(A):
    b = np.array([0.0, 0.0, 1.0])
    affine = np.insert(A, 2, 0.0, axis=1)
    affine = np.insert(affine, 2, 0.0, axis=0)
    affine[2][2] = 1.0
    #print("AffineMatrix 3x3:\n ", affine, "\n")
    return affine

#LEFT MATRIX
def create_left_matrix(wW, wH):
    a = [1.0, 0.0, wW / 2.0]
    b = [0.0, 1.0, wH / 2.0]
    c = [0.0, 0.0, 1.0]
    M = np.vstack([a, b, c])
    #print("Left Matrix:\n ", M, "\n")
    return M

#RIGHT MATRIX
def create_right_matrix(rW, rH):
    a = [1.0, 0.0, -1.0 * (rW / 2.0)]
    b = [0.0, 1.0, -1.0 * (rH/ 2.0)]
    c = [0.0, 0.0, 1.0]
    M = np.vstack([a, b, c])
    #print("Right Matrix:\n ", M, "\n")
    return M


L = create_left_matrix(wH, wW)
Affine3x3 = create_3x3_affine(A)
R = create_right_matrix(rH, rW)

H = np.dot(L, Affine3x3)
H = np.dot(H, R) # H = L * Affine * R
#print("Homography Matrix = LeftMatrix * Affine 3x3 Matrix * RightMatrix\n ", H, "\n")

def apply_bilinear_interpolation(img, point):
    x = point[0]
    y = point[1]

    if(int(y) == 799): #because of out of bound error in some edges
        y = y - 1.0
    if(int(x) == 639):
        x = x - 1.0
    alfa =  abs(np.round(x) - x)
    beta =  abs(np.round(y) - y)

    x1 = int(np.round(x))
    y1 = int(np.round(y))

    if ((x1 == int(x)) | (y1 == int(y)) ):
        I1 = img[x1][y1] * (1 - alfa) * (1 - beta)
        I2 = img[int(x)+1][y1] * (alfa) * (1 - beta)
        I3 = img[x1][int(y)+1] * (1 - alfa) * (beta)
        I4 = img[int(x)+1][int(y)+1] * (alfa) * (beta)

    else:
        I1 = img[x1][y1] * (1 - alfa) * (1 - beta)
        I2 = img[int(x)][y1] * (alfa) * (1 - beta)
        I3 = img[x1][int(y)] * (1 - alfa) * (beta)
        I4 = img[int(x)][int(y)] * (alfa) * (beta)
    return I1 + I2 + I3 + I4  #resulting intensity


def find_homography(img, warped, H):
    inv = np.linalg.inv(H)
    # each pixel at reference image first transformed with H, and then transformed with inverse H, and sending intensity to output image
    for row in range(0, len(img)):
        for col in range(0, len(img[0])):

            xy_homo_coor = np.array([[row],[col],[1.0]])

            new_coor = np.dot(H, xy_homo_coor)

            warpedX = int(np.round(new_coor[0]))
            warpedY = int(np.round(new_coor[1]))

            if (np.round(warpedX) == len(warped)):
                warpedX = warpedX - 1
            if (np.round(warpedY) == len(warped[0])):
                warpedY = warpedY - 1

            final_coor = np.dot(inv, new_coor)

            final_intensity = apply_bilinear_interpolation(img, final_coor)

            warped[warpedX][warpedY] = final_intensity

    return warped


temp = np.zeros((wH, wW)) #create black image wH x wW

temp = find_homography(img, temp, H) #find intensities of that image

cv2.imwrite("FinalImageWithBilinearInterpolation.png", temp)

print("\nWarped image has been constructed succesfully!")


'''
     COMMENT-1

When I multiplied Affine matrix from left side, center of the image has been moved to right bottom side,(also gave the out of bound error)
and when I multiplied Affine matrix from right side, center of image has been moved to left top side,
to find correct position of the warped image center, we should multiply from both side.

The reason behind that is: After multiplying Affine with Right matrix,resulting Homography matrix has negative 
translation values tx and ty, and when we multiply original image coordinates with that H,
image pixel coordinates has been translated with these negative tx and ty values.So, center has moved to top left.

After multiplying Affine with Left matrix,resulting Homography matrix has positive 
translation values tx and ty, and when we multiply original image coordinates with that H,
image pixel coordinates has been translated with these positive tx and ty values.So, center has moved to bottom right.

Finally, if we multiply with both of them, we find correct positions of pixel coordinates like in Figure 2 in homework pdf.
     
I added the resulting image which has H = Affine*Right, and image has moved to top left, when array indices goes negative, pixels has gone
top right, bottom left and bottom- right,then resulting image has become like 'RightxAffine.png'.

L x Affine gives error becase we exceed size of image, so I could not added its resulting image.
     
     COMMENT-2

Firstly we transformed our reference image with Homography matrix to find corresponding locations of 
pixels of reference image.At that point,When I printed the corrensponding coordinates of reference pixels,
some of the pixels has the same corresponding coordinates, and this is a problem.What is the correct intensity of
that corresponding pixels? Inverse homography gives us the actual location of that corresponding points in reference image.

To find actual intensity values of that corresponding pixel,we multiply its coordinates with inverse Homography matrix,
then we apply interpolation to corresponding pixel at reference image, finally we find actual intensity of transformed point.


     COMMENT-3
     
As I mentioned in COMMENT-2, we find corresponding point in reference image with inverse Homography,
if that corresponding points are between some pixels, we cannot find its intensity correctly without interpolation.
We use bilinear interpolation because in bilinear interpolation, if a corresponding point has non-integer x or y values,
for example, (0.6, 1.7) , all 4 nearest pixel intensities contributes to intensity value of pixel at (0.6,1.7).If one of the 4 points
has a very big different intensity from others, this can give undesired intensity to us and at that point nearest neighbour method
is better.I think, in our image, bilinear interpolation is better because when I look at the intensity of images, I see intensities
are distributed closely and contribution of 4 neighbour is better than one in that case.In a regions which a lot of pixels
has the same intensity, there is no difference between bilinear and nearest neighbour interpolation.

I added results with two interpolations, and there is no big difference between them.
My observation is that bilinear interpolation is better in terms of reducing noise because it takes weighted sum of 4 neighbour,
and nearest neighbour is better in preserving edges, in bilinear interpolation result are worse in edges because of intensity differences.


'''








