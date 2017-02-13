# perspective transform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

img=mpimg.imread('./output_images/frame347_undist.jpg')

offset=350
# FOUR SOURCE COORDINATES
src = np.float32(
    [[328, 670],
    [1081, 670],
    [601, 450],
    [687, 450]])

# FOUR DESTINATION COORDINATES
dst = np.float32(
    [
    [src[0][0], img.shape[0]],
    [src[1][0], img.shape[0]],
    [src[0][0], 0],
    [src[1][0], 0]
    ])

# better
dst = np.float32(
    [
    [offset, img.shape[0]],
    [img.shape[1]-offset, img.shape[0]],
    [offset, 0],
    [img.shape[1]-offset, 0]
    ])


plt.figure(1)
#original
plt.subplot(121)
plt.imshow(img)
print ()
plt.plot([src[0][0],src[1][0],src[3][0],src[2][0],src[0][0]],
    [src[0][1],src[1][1],src[3][1],src[2][1],src[0][1]])
plt.title('Original straight road')

img_size = (img.shape[1], img.shape[0])

#perspective transformed
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

plt.subplot(122)
plt.imshow(warped)
plt.plot([dst[0][0],dst[1][0],dst[3][0],dst[2][0],dst[0][0]],
    [dst[0][1],dst[1][1],dst[3][1],dst[2][1],dst[0][1]])
plt.title('Warped')

plt.show()
cv2.imwrite('./output_images/frame347_warped.jpg',warped)
