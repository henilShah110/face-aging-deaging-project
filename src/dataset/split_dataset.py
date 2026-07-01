import os
import shutil

# creating directories of faces,young and old in data/processed

def split_by_age(input_dir, output_dir):
    young_dir = os.path.join(output_dir, "young")
    old_dir = os.path.join(output_dir, "old")

    os.makedirs(young_dir, exist_ok=True)
    os.makedirs(old_dir, exist_ok=True)

    young_count = 0
    old_count = 0

    for file in os.listdir(input_dir):
        try:
            age = int(file.split("_")[1].split(".")[0])

            src_path = os.path.join(input_dir, file)

            if age <= 25:
                dst_path = os.path.join(young_dir, file)
                shutil.copy2(src_path, dst_path)
                young_count += 1

            elif age >= 50:
                dst_path = os.path.join(old_dir, file)
                shutil.copy2(src_path, dst_path)
                old_count += 1

        except:
            continue

    print("Dataset split complete!")
    print(f"Young images: {young_count}")
    print(f"Old images: {old_count}")