import cv2
import os
from config import WIKI_IMAGE_DIR, PROCESSED_FACE_DIR


def crop_and_save(sample, idx):
    img_path = os.path.join(WIKI_IMAGE_DIR, sample["path"])
    img = cv2.imread(img_path)

    if img is None:
        return False

    bbox = sample["bbox"]

    # flatten in case it's nested
    bbox = bbox.flatten()

    x1, y1, x2, y2 = map(int, bbox)

    face = img[y1:y2, x1:x2]

    if face.size == 0:
        return False

    face = cv2.resize(face, (128, 128))

    save_path = os.path.join(PROCESSED_FACE_DIR, f"{idx}_{sample['age']}.jpg")
    cv2.imwrite(save_path, face)

    return True