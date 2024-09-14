import cv2

#Load the image 

img = cv2.imread('./basic-cv/image.png')

#Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Display grayscale image
cv2.imshow('Grayscale Image', gray_img)
cv2.waitKey(5000)
cv2.destroyAllWindows()