import cv2
import os
import numpy as np
import main as face_reco

def test_face_recognizer(model_path, test_data_path):
    # Load the trained face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    try:
        face_recognizer.read(model_path)
    except cv2.error as e:
        print(f"Error: Failed to load the face recognizer model from '{model_path}'.")
        print(f"OpenCV Error: {e}")
        return

    # Initialize variables for accuracy calculation
    total_faces = 0
    correctly_predicted_faces = 0

    # Load dynamic label mapping
    label_mapping_path = os.path.join(os.path.dirname(model_path), "label_mapping.npy")
    try:
        label_mapping = np.load(label_mapping_path, allow_pickle=True).item()
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"Error: Failed to load dynamic label mapping from '{label_mapping_path}'.")
        print(f"Error Details: {e}")
        return

    # Iterate through the test data
    for person_name in os.listdir(test_data_path):
        person_path = os.path.join(test_data_path, person_name)

        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_path, filename)
                    test_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if test_img is not None:
                        # Detect faces in the test image
                        faces_detected, _ = face_reco.faceDetection(test_img)

                        if faces_detected is not None and len(faces_detected) > 0:
                            total_faces += 1

                            # Assume the ground truth label is the same as the person's name
                            try:
                                ground_truth_label = label_mapping[person_name]
                            except KeyError:
                                print(f"Error: Person '{person_name}' not found in dynamic label mapping. Skipping...")
                                continue

                            # Process each detected face
                            for face in faces_detected:
                                (x, y, w, h) = face
                                roi_gray = test_img[y:y + w, x:x + h]

                                # Predict the label using the face recognizer
                                predicted_label, _ = face_recognizer.predict(roi_gray)

                                # Check if the prediction matches the ground truth
                                if predicted_label == ground_truth_label:
                                    correctly_predicted_faces += 1

    # Calculate accuracy
    accuracy = (correctly_predicted_faces / total_faces) * 100 if total_faces > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")

# Example usage
model_path = "/home/elshorbagy/Desktop/face_reco/out/out.yml"
test_data_path = "/home/elshorbagy/Desktop/face_reco/test_data"

test_face_recognizer(model_path, test_data_path)
