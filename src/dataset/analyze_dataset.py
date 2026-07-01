import os
from collections import Counter
import matplotlib.pyplot as plt

# to check the age range of images we have using min age
# and max age and creating a histogram


def analyze_age_distribution(face_dir):
    ages = []

    for file in os.listdir(face_dir):
        try:
            age = int(file.split("_")[1].split(".")[0])
            ages.append(age)
        except:
            continue

    print(f"Total images: {len(ages)}")
    print(f"Min age: {min(ages)}")
    print(f"Max age: {max(ages)}")

    counter = Counter(ages)

    plt.figure(figsize=(10, 5))
    plt.bar(counter.keys(), counter.values())
    plt.xlabel("Age")
    plt.ylabel("Number of Images")
    plt.title("Age Distribution")
    plt.show()