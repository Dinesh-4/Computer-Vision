import cv2
import cv2.data

#Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


#Load the image and convert it to grayscale
img = cv2.imread('./basic-cv/people.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=5.1, minNeighbors=6)

#Draw rectangles around the faces
for(x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 9)

#Display the image with detected faces
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()