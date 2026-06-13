import dlib 
import numpy as np
import face_recognition_models
from sklearn.svm import SVC 
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import streamlit as st
from src.database.db import get_all_students

@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector() 

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec


def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, 1)

    encodings = []

    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1) # 128 embedding
        encodings.append(np.array(face_descriptor))

    return encodings


@st.cache_resource
def get_trained_model():
    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None
    
    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))

    if len(X) == 0:
        return None  # Changed from 0 to None for consistent dictionary parsing
    
    # Extract unique classes
    unique_classes = np.unique(y)
    
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')

    # SVM can only train if there are 2 or more distinct student IDs
    if len(unique_classes) >= 2:
        try:
            clf.fit(X, y)
        except ValueError:
            # If fitting fails, ensure we do not return an unfitted model
            return {'clf': None, 'X': X, 'y': y}
        return {'clf': clf, 'X': X, 'y': y}
    else:
        # 0 or 1 student: Return None for clf so the pipeline uses 1-NN fallback
        return {'clf': None, 'X': X, 'y': y}


def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    # Ensure model data exists and a valid classifier structure returned
    return bool(model_data)


def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)
    detected_student = {}

    model_data = get_trained_model()

    if not model_data:
        return detected_student, [], len(encodings)
    
    clf = model_data["clf"]
    X_train = model_data['X']
    y_train = model_data['y']

    all_students = sorted(list(set(y_train)))

    # Defensive check: Ensure clf is not None and actually fitted
    is_model_ready = False
    if clf is not None:
        try:
            check_is_fitted(clf)
            is_model_ready = True
        except NotFittedError:
            is_model_ready = False

    for encoding in encodings:
        # Use SVM prediction only if model is fully trained and ready
        if is_model_ready and len(all_students) >= 2:
            try:
                predicted_id = int(clf.predict([encoding])[0])
            except Exception:
                # Dynamic fallback to 1-Nearest Neighbor Euclidean distance if prediction breaks
                distances = [np.linalg.norm(np.array(emb) - encoding) for emb in X_train]
                predicted_id = int(y_train[np.argmin(distances)])
        elif len(all_students) > 0:
            # Fallback for 1 single registered student
            predicted_id = int(all_students[0])
        else:
            continue

        # Find the stored embedding for the predicted student to check threshold
        try:
            student_embedding = X_train[y_train.index(predicted_id)]
            best_match_score = np.linalg.norm(student_embedding - encoding)
            
            resemblance_threshold = 0.6
            if best_match_score <= resemblance_threshold:
                detected_student[predicted_id] = True
        except ValueError:
            continue
            
    return detected_student, all_students, len(encodings)
