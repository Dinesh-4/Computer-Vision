import cv2
import imutils

# Start video capture from the webcam
cam = cv2.VideoCapture(0)
firstFrame = None
area = 500

while True:
    ret, img = cam.read()

    # Check if the frame was read properly
    if not ret:
        print("Error: Couldn't read frame from the camera.")
        break

    # Resize the image to a fixed width for consistency
    img = imutils.resize(img, width=1000)

    # Convert the frame to grayscale
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and smooth the image
    gaussianImg = cv2.GaussianBlur(grayimg, (21, 21), 0)

    # If this is the first frame, initialize it and skip the rest of the loop
    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    # Compute the absolute difference between the first frame and the current frame
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)

    # Apply threshold to binarize the image, isolating moving objects
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the threshold image to fill in small holes
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    # Find contours in the thresholded image
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Loop over the contours
    for c in cnts:
        # Ignore small contours below the defined area threshold
        if cv2.contourArea(c) < area:
            continue

        # Compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "moving object detected"

    # Display the status message on the frame
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the video feed with bounding boxes and text
    cv2.imshow("camerafeed", img)

    # Exit the loop if the 'q' key is pressed
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
