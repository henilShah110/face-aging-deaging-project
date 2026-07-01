import scipy.io
import numpy as np
from datetime import datetime
from config import WIKI_MAT_PATH


def load_mat_file():
    """
    Loads the .mat file and extracts required fields
    """

    # Load the .mat file (MATLAB format)
    data = scipy.io.loadmat(WIKI_MAT_PATH)

    # 'wiki' is the main struct inside the file
    wiki = data['wiki']

    # Extract fields (VERY nested structure)
    dob = wiki['dob'][0][0][0]
    photo_taken = wiki['photo_taken'][0][0][0]
    full_path = wiki['full_path'][0][0][0]
    face_score = wiki['face_score'][0][0][0]
    second_face_score = wiki['second_face_score'][0][0][0]
    face_location = wiki['face_location'][0][0][0]

    return dob, photo_taken, full_path, face_score, second_face_score, face_location


def clean_data(dob, photo_taken, full_path, face_score, second_face_score, face_location):
    """
    Cleans dataset and returns usable entries
    """

    clean_samples = []

    for i in range(len(dob)):
        try:
            # ---- 1. FILTER BAD FACES ----

            # Remove low confidence detections
            if face_score[i] < 1.0:
                continue

            # Remove images with multiple faces
            if not np.isnan(second_face_score[i]):
                continue

            # ---- 2. CALCULATE AGE ----

            # Convert MATLAB date to Python year
            birth_year = datetime.fromordinal(int(dob[i])).year

            # Age = photo year - birth year
            age = int(photo_taken[i]) - int(birth_year)

            # Remove unrealistic ages
            if age < 0 or age > 100:
                continue

            # ---- 3. EXTRACT PATH ----

            img_path = full_path[i][0]

            # ---- 4. EXTRACT BOUNDING BOX ----

            bbox = face_location[i]  # [x1, y1, x2, y2]

            # ---- 5. STORE CLEAN SAMPLE ----

            clean_samples.append({
                "path": img_path,
                "age": age,
                "bbox": bbox
            })

        except Exception as e:
            # Skip broken entries
            continue

    return clean_samples