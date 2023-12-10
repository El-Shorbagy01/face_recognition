import cv2

cpt = 0  # Counter for image filenames

# Open a connection to the webcam (0 represents the default camera)
vidStream = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = vidStream.read()

    # Display the captured frame
    cv2.imshow("Test Frame", frame)

    # Save the captured frame as an image with an incrementing filename
    cv2.imwrite('/home/elshorbagy/Desktop/face_reco/data/elshorbagy/image%04i.jpg' % cpt, frame)
    
    # Increment the counter
    cpt += 1

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release the webcam and close the OpenCV window
vidStream.release()
cv2.destroyAllWindows()
