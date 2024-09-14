import cv2

#Initialize video capture (0 is for the default camera )
cap = cv2.VideoCapture(0)

#Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)


while True:

    #Capture frame-by-frame
    ret, frame = cap.read()
    
     # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    #Apply background subtraction to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # cv2.imshow("before noise ", fg_mask)
    #Removing noise and small objects using morphological operation (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    #Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    #Draw bounding boxes around detected moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            text = "Moving object detected"


    #Display the status message on the frame 
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    #Display the original frame with bounding boxes
    cv2.imshow('Moving Object Detection', frame)

    #Display the foreground mask (for visualization purposes)
    cv2.imshow('Foreground Mask', fg_mask)


    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()