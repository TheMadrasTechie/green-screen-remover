import cv2
import numpy as np
import skimage.exposure

# load image
img = cv2.imread('demo.jpg')

# convert to LAB
lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

# extract A channel
A = lab[:,:,1]

# threshold A channel
thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# blur threshold image
blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT)

# stretch so that 255 -> 255 and 127.5 -> 0
mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255)).astype(np.uint8)

# add mask to image as alpha channel
result = img.copy()
result = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
result[:,:,3] = mask

# save output
cv2.imwrite('greenscreen_thresh.png', thresh)
cv2.imwrite('greenscreen_mask.png', mask)
cv2.imwrite('greenscreen_antialiased.png', result)

# Display various images to see the steps
cv2.imshow('A',A)
cv2.imshow('thresh', thresh)
cv2.imshow('blur', blur)
cv2.imshow('mask', mask)
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()