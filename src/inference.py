import torch
import cv2
import numpy as np

from PIL import Image
import torchvision.transforms as transforms

from src.models.generator import Generator
from src.utils.face_detection import detect_face

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# NORMAL TRANSFORMING
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

def load_generator(
    checkpoint_path,
    mode
):
    generator = Generator().to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if mode == "age":
        generator.load_state_dict(
            checkpoint["G_YO"]
        )
    else:
        generator.load_state_dict(
            checkpoint["G_OY"]
        )
    generator.eval()

    return generator

def process_image(
    image,
    generator
):
    face, bbox = detect_face(image)

    if face is None:
        return None, None, False
    
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    face = Image.fromarray(face)

    input_tensor = transform(face)
    input_tensor = input_tensor.unsqueeze(0) #unsqueeze to add imaginary batch size
    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        output = generator(
            input_tensor
        )
        output = output.squeeze(0)
        output = (output + 1) / 2
        generated_face = (
            output
            .permute(1,2,0)
            .cpu()
            .numpy()
        )
        generated_face = (
            generated_face * 255
        ).astype(np.uint8)

        generated_face = cv2.cvtColor(
            generated_face,
            cv2.COLOR_RGB2BGR
        )
        x, y, w, h = bbox

        generated_face = cv2.resize(
            generated_face,
            (w,h),
            interpolation=cv2.INTER_CUBIC
        )

        mask = 255 * np.ones(
            generated_face.shape,
            dtype=np.uint8
        )

        center = (
            x + w//2,
            y + h//2
        )

        result = cv2.seamlessClone(
            generated_face,
            image,
            mask,
            center,
            cv2.NORMAL_CLONE
        )

    return result, generated_face, True
