import cv2

#Load the image
img = cv2.imread('./basic-cv/people.jpg')

#Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Perform Canny edge detection
edges = cv2.Canny(gray_img, threshold1=100, threshold2=100)

#Display edge-detected image
cv2.imshow('Edge Detection', edges)
cv2.waitKey(5000)
cv2.destroyAllWindows()