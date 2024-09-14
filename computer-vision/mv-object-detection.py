import cv2

#Initialize video capture (0 is for the default camera )
cap = cv2.VideoCapture(0)

#Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)


while True:

    #Capture frame-by-frame
    ret, frame = cap.read()
    
     # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    #Apply background subtraction to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)


    #Removing noise and small oject


    cv2.imshow('Foreground Mask', fg_mask)


    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()