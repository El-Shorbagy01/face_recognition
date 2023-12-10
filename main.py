import numpy as np
import cv2
import os

def faceDetection(img):
    # Check if img is a NumPy array or a scalar
    if isinstance(img, np.ndarray) or np.isscalar(img):
        if img.ndim == 3:  # Check if the image has three channels (BGR)
            # Convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:  # Check if the image is already grayscale
            gray_img = img
        else:
            print(f"Error: Unsupported number of channels in the image ({img.ndim}).")
            return None, None
    else:
        print(f"Error: Unexpected type for img. Got {type(img)}, expected numpy array or scalar.")
        return None, None
    
    # Load the face classifier
    face_haar = cv2.CascadeClassifier('/home/elshorbagy/Desktop/face_reco/haarcascade_frontalface_alt.xml')
    
    # Detect faces in the grayscale image
    faces = face_haar.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=3)
    
    return faces, gray_img

def labels_for_training_data(directory):
    data = []

    for person_id, person_name in enumerate(os.listdir(directory)):
        person_path = os.path.join(directory, person_name)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_path, filename)
                    # Extract label dynamically from folder name
                    label = person_id
                    test_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if test_img is None:
                        print("Error: Unable to load image from", image_path)
                        continue

                    faces_rect, gray_img = faceDetection(test_img)
                    if len(faces_rect) == 0:
                        print("Error detecting faces in image:", image_path)
                        continue

                    print("Faces detected:", faces_rect)
                    (x, y, w, h) = faces_rect[0]
                    roi_gray = gray_img[y:y + w, x:x + h]
                    data.append((roi_gray, label))

    return data
 

def train_classifier(faces, face_labels):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(face_labels))
    return face_recognizer

# Draw a rectangle on the face function
def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)

# Putting text on images
def put_text(test_img, label_name, x, y):
    cv2.putText(test_img, label_name, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
