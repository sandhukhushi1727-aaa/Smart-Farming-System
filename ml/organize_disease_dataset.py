import os
import random
import shutil
from pathlib import Path

# Original PlantVillage dataset location
SOURCE_DIR = Path("datasets/plant_disease/plantvillage dataset/color")

# Destination folder
DEST_DIR = Path("datasets/plant_disease")

# Five tomato disease classes
classes = {
    "Tomato___healthy": "healthy",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___Septoria_leaf_spot": "septoria_leaf_spot"
}

# Split percentages
train_ratio = 0.70
validation_ratio = 0.15

random.seed(42)

# Supported image formats
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

for original_class, new_class in classes.items():

    source_folder = SOURCE_DIR / original_class

    if not source_folder.exists():
        print("Folder not found:", source_folder)
        continue

    images = [
        file for file in source_folder.iterdir()
        if file.is_file() and file.suffix.lower() in image_extensions
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    splits = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():

        destination_folder = DEST_DIR / split_name / new_class

        destination_folder.mkdir(parents=True, exist_ok=True)

        for image_file in split_images:
            destination = destination_folder / image_file.name

            shutil.copy2(image_file, destination)

    print(f"\n{new_class}")
    print("Total images:", total)
    print("Training:", len(train_images))
    print("Validation:", len(validation_images))
    print("Testing:", len(test_images))

print("\nDataset organization completed!")