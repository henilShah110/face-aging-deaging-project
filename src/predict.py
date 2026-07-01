import argparse
import os
import time

import cv2

from src.inference import (
    load_generator,
    process_image,
    device
)


# PARSING - To create CLI
parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir",
    default="test_images"
)

parser.add_argument(
    "--output_dir",
    default="outputs"
)

parser.add_argument(
    "--checkpoint",
    default="checkpoints/latest.pth"
)

parser.add_argument(
    "--mode",
    choices=["age", "deage"],
    default="age"
)

args = parser.parse_args()

if not os.path.exists(args.input_dir):
    raise FileNotFoundError(
        f"Input directory '{args.input_dir}' does not exist."
    )

# output folder
os.makedirs(
    args.output_dir,
    exist_ok=True
)

generator = load_generator(
    args.checkpoint,
    args.mode
)

print(f"Using device: {device}")
print(f"Loaded checkpoint: {args.checkpoint}")
print(f"Mode: {args.mode}")
print("Generator loaded successfully.")


# processing images
image_files = [

    file

    for file in os.listdir(args.input_dir)

    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )

]

processed = 0
failed = 0

for filename in image_files:

    print(f"Processing {filename}...")

    # reading the image
    image_path = os.path.join(
        args.input_dir,
        filename
    )

    original = cv2.imread(
        image_path
    )

    if original is None:
        print(
            f"Could not open {filename}"
        )
        failed += 1
        continue

    start = time.time()

    result, generated_face, success = process_image(
        original,
        generator
    )
    #print(type(result))
    #print(type(generated_face))
    #print(success)

    end = time.time()
    print(
        f"Inference time: {end-start:.3f} sec"
    )

    if not success:
        print(
            f"No face detected in {filename}"
        )
        failed += 1

        continue

    # Save final result
    output_path = os.path.join(
        args.output_dir,
        filename
    )

    cv2.imwrite(   
        output_path,
        result
    )

    # Save comparison image
    comparison = cv2.hconcat([
        original,
        result
    ])

    comparison_path = os.path.join(
        args.output_dir,
        f"comparison_{filename}"
    )

    cv2.imwrite(
        comparison_path,
        comparison
    )

    face_output = os.path.join(
        args.output_dir,
        f"generated_face_{filename}"
    )

    cv2.imwrite(
        face_output,
        generated_face
    )
    
    print(
        f"Saved result: {output_path}"
    )
    print(
        f"Saved comparison: {comparison_path}"
    )
    processed+=1
    print("-" * 50)

print("\nFinished!")
print(f"Processed : {processed}")
print(f"Failed    : {failed}\n")