import cv2

#Load the image 
img = cv2.imread('./basic-cv/image.png')

#Display the image
cv2.imshow('Image Window', img)


#Wait for a key event and close the window
cv2.waitKey(5000)
cv2.destroyAllWindows()