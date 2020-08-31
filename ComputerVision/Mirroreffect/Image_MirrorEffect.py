import numpy as np
import matplotlib.pyplot as plt
import cv2

%matplotlib inline

img = cv2.imread('ProfilePic.jpeg')

plt.imshow(img[:,:,::-1])
plt.show()

img.shape

imgNew = np.zeros((381,692,3), dtype=np.uint8)

plt.imshow(imgNew[:,:,::-1])
plt.show()

imgNew[:,:346] = img

plt.imshow(imgNew[:,:,::-1])
plt.show()

imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

hue,sat,val = cv2.split(imgHSV)

plt.imshow(val, cmap="gray")
plt.show()

val = 255 - val

plt.imshow(val, cmap="gray")
plt.show()

imgHSV = cv2.merge((val,val,val))

plt.imshow(imgHSV[:,:,::-1])
plt.show()

imgHSV = imgHSV[:,::-1,:]

plt.imshow(imgHSV[:,:,::-1])
plt.show()

imgNew[:,346:,:] = imgHSV
plt.imshow(imgNew[:,:,::-1])
plt.show()

img = img[:,::-1,:]

plt.imshow(img[:,:,::-1])
plt.show()

imgNew[:,346:,:] = img
plt.imshow(imgNew[:,:,::-1])
plt.show()

cv2.imwrite("mirror_effect_profile.jpg", imgNew)
