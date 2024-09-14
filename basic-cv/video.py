import cv2

#Start video capture
cap = cv2.VideoCapture(0)


while True:
    #Capture frame-by-frame
    ret, frame = cap.read()

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    #display the frame
    cv2.imshow('Video Stream', frame)

    #Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the capture and close the window

cap.release()
cv2.destroyAllWindows()