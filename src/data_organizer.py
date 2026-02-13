import os
import shutil
from pathlib import Path
import random

def organize_data(source, destination, split = 0.8):
    source_path = Path(source)
    destination_path = Path(destination)

    # collect the images for cats and dogs
    cats = sorted(source_path.glob("cat.*.jpg"))
    dogs = sorted(source_path.glob("dog.*.jpg"))

    print(f"Found {len(cats)} cats images & {len(dogs)} dogs images.")

    # shuffle the images
    random.seed(1)
    random.shuffle(cats)
    random.shuffle(dogs)

    # calculate split index
    split_cats = int(len(cats) * split)
    split_dogs = int(len(dogs) * split)

    # split based on the index and provided split matrix
    training_cats = cats[:split_cats]
    testing_cats = cats[split_cats:]
    training_dogs = dogs[:split_dogs]
    testing_dogs = dogs[split_dogs:]

    print("\nSplitting data...")
    print(f"Training Data: {len(training_cats)} cats, {len(training_dogs)} dogs")
    print(f"Test Data: {len(testing_cats)} cats, {len(testing_dogs)} dogs")

    # copying files
    print("\nCopying files...")
    for image in training_cats:
        path = destination_path / "train" / "cats" / image.name
        shutil.copy2(image, path)

    for image in training_dogs:
        path = destination_path / "train" / "dogs" / image.name
        shutil.copy2(image, path)

    for image in testing_cats:
        path = destination_path / "test" / "cats" / image.name
        shutil.copy2(image, path)

    for image in testing_dogs:
        path = destination_path / "test" / "dogs" / image.name
        shutil.copy2(image, path)

    print("\nSuccessfully copied data to the selected directories!\n")

    # Final verification
    print("\nFinal Structure:")
    print(f"train/cats: {len(list((destination_path / "train" / "cats").glob('*.jpg')))} images")
    print(f"train/dogs: {len(list((destination_path / "train" / "dogs").glob('*.jpg')))} images")
    print(f"test/cats: {len(list((destination_path / "test" / "cats").glob('*.jpg')))} images")
    print(f"test/dogs: {len(list((destination_path / "test" / "dogs").glob('*.jpg')))} images")

if __name__ == "__main__":
    source = "/home/tahsinttalha/MLcode/cats-vs-dogs-classification/unorganised_data"   # only change this to your source path
    destination = "/home/tahsinttalha/MLcode/cats-vs-dogs-classification/data"          # only change this to your destination path

    organize_data(source, destination, split=0.8)