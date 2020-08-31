import numpy as np;
import cv2;
import matplotlib.pyplot as plt;

%matplotlib inline

img=cv2.imread('ProfilePic.jpeg')
plt.imshow(img[:,:,::-1])
plt.show()

img.shape

imgNew = np.zeros((762,346,3),dtype=np.uint8)
plt.imshow(imgNew[:,:,::-1])
plt.show()

imgNew[:381][:] = img

plt.imshow(imgNew[:,:,::-1])
plt.show()

imgInverted = img[::-1,:,:]

plt.imshow(imgInverted[:,:,::-1])
plt.show()

imgNew[381:][:] = imgInverted

plt.imshow(imgNew[:,:,::-1])
plt.show()

cv2.imwrite("water_effect_profile.jpg", imgNew)