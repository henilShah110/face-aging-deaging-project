import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection

FACE_PADDING = 0.20
MIN_DETECTION_CONFIDENCE = 0.5
MODEL_SELECTION = 0

detector = mp_face.FaceDetection(
    model_selection=MODEL_SELECTION,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE
)


def detect_face(image):
    """
    Detect the largest face in an image.

    Parameters
    ----------
    image : np.ndarray (OpenCV BGR image)

    Returns
    -------
    face_crop : np.ndarray
        Cropped face image

    bbox : tuple
        (x, y, w, h)

    Returns (None, None) if no face is found.
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = detector.process(rgb)

    if not results.detections:
        return None, None

    h, w, _ = image.shape

    # If multiple faces exist, choose the biggest one
    best_detection = max(
        results.detections,
        key=lambda d:
        d.location_data.relative_bounding_box.width *
        d.location_data.relative_bounding_box.height
    )

    box = best_detection.location_data.relative_bounding_box

    x = int(box.xmin * w)
    y = int(box.ymin * h)
    bw = int(box.width * w)
    bh = int(box.height * h)

    x_pad = int(bw * FACE_PADDING)
    y_pad = int(bh * FACE_PADDING)

    # Expand bounding box
    x -= x_pad
    y -= y_pad
    bw += 2 * x_pad
    bh += 2 * y_pad

    # Clip to image boundaries
    x = max(0, x)
    y = max(0, y)

    bw = min(bw, w - x)
    bh = min(bh, h - y)

    face_crop = image[
        y:y + bh,
        x:x + bw
    ]

    return face_crop, (x, y, bw, bh)