import numpy as np
import cv2
import os
import main as face_reco

# Training
dataset_path = "/home/elshorbagy/Desktop/face_reco/data"
training_data = face_reco.labels_for_training_data(dataset_path)

faces, face_labels = zip(*training_data)  # Unpack the list of tuples

face_recognizer = face_reco.train_classifier(faces, face_labels)
face_recognizer.write("/home/elshorbagy/Desktop/face_reco/out/out.yml")

# Dynamic label mapping during training
label_mapping = {label: f'Person{label + 1}' for label in set(face_labels)}

# save dynamic label mapping
np.save(os.path.join('/home/elshorbagy/Desktop/face_reco/out/', 'label_mapping.npy'), label_mapping)

# Testing
test_img = cv2.imread("/home/elshorbagy/Desktop/face_reco/data/Aaron_Guiel/Aaron_Guiel_0001.jpg")

if test_img is not None:
    faces_detected, gray_img = face_reco.faceDetection(test_img)

    if faces_detected is not None:
        # Confidence threshold for predictions
        confidence_threshold = 10 # Adjust as needed

        for face in faces_detected:
            (x, y, w, h) = face
            roi_gray = gray_img[y:y + w, x:x + h]
            label, confidence = face_recognizer.predict(roi_gray)

            # Use dynamic label mapping
            predict_name = label_mapping.get(label, "Unknown")

            # Check confidence level
            if confidence < confidence_threshold:
                predict_name = "Unknown"

            print("Predicted Name:", predict_name)
            print("Confidence:", confidence)

            face_reco.draw_rect(test_img, face)
            face_reco.put_text(test_img, predict_name, x, y)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Detected Faces', resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No faces detected in the test image.")
else:
    print("Error: Unable to load the test image.")
