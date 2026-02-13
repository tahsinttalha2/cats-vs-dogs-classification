import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def explore_dataset(data_dir):
    """
    Explore and visualize the dataset
    """
    data_path = Path(data_dir)
    
    # Get counts
    train_cats = list((data_path / "train" / "cats").glob("*.jpg"))
    train_dogs = list((data_path / "train" / "dogs").glob("*.jpg"))
    test_cats = list((data_path / "test" / "cats").glob("*.jpg"))
    test_dogs = list((data_path / "test" / "dogs").glob("*.jpg"))
    
    print("=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"\nTraining Set:")
    print(f"  Cats: {len(train_cats):,} images")
    print(f"  Dogs: {len(train_dogs):,} images")
    print(f"  Total: {len(train_cats) + len(train_dogs):,} images")
    
    print(f"\nTest Set:")
    print(f"  Cats: {len(test_cats):,} images")
    print(f"  Dogs: {len(test_dogs):,} images")
    print(f"  Total: {len(test_cats) + len(test_dogs):,} images")
    
    print(f"\n📊 Class Balance:")
    print(f"  Training: {len(train_cats)/(len(train_cats)+len(train_dogs))*100:.1f}% cats, {len(train_dogs)/(len(train_cats)+len(train_dogs))*100:.1f}% dogs")
    
    # Randomly sample some images to examine
    print("\n" + "=" * 50)
    print("IMAGE PROPERTIES")
    print("=" * 50)
    
    sample_images = random.sample(train_cats + train_dogs, 5)
    
    sizes = []
    for img_path in sample_images:
        img = Image.open(img_path)
        width, height = img.size
        sizes.append((width, height))
        print(f"\n{img_path.name}:")
        print(f"  Size: {width} x {height} pixels")
        print(f"  Mode: {img.mode} (RGB = 3 channels)")
        print(f"  Format: {img.format}")
    
    # Check if all images are the same size
    unique_sizes = set(sizes)
    print(f"\n🔍 Unique image sizes found: {len(unique_sizes)}")
    if len(unique_sizes) > 1:
        print("⚠️  Images have different sizes - we'll need to resize them!")
        print(f"   Size variations: {unique_sizes}")
    else:
        print(f"✅ All images are the same size: {sizes[0]}")
    
    # Visualize some images
    visualize_samples(train_cats, train_dogs, num_samples=8)

def visualize_samples(cat_images, dog_images, num_samples=8):
    """
    Display a grid of sample images
    """
    print("\n" + "=" * 50)
    print("VISUALIZING SAMPLES")
    print("=" * 50)
    
    # Sample equal number from cats and dogs
    samples_per_class = num_samples // 2
    
    cat_samples = random.sample(cat_images, samples_per_class)
    dog_samples = random.sample(dog_images, samples_per_class)
    
    # Create figure
    fig, axes = plt.subplots(2, samples_per_class, figsize=(15, 6))
    fig.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold')
    
    # Plot cats (top row)
    for idx, img_path in enumerate(cat_samples):
        img = Image.open(img_path)
        axes[0, idx].imshow(img)
        axes[0, idx].axis('off')
        axes[0, idx].set_title(f'Cat\n{img.size[0]}x{img.size[1]}', fontsize=10)
    
    # Plot dogs (bottom row)
    for idx, img_path in enumerate(dog_samples):
        img = Image.open(img_path)
        axes[1, idx].imshow(img)
        axes[1, idx].axis('off')
        axes[1, idx].set_title(f'Dog\n{img.size[0]}x{img.size[1]}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=150, bbox_inches='tight')
    print("\n✅ Visualization saved as 'data_exploration.png'")
    print("   (Check your project root directory)")
    
    # Don't call plt.show() in case running in non-interactive environment
    # plt.show()

if __name__ == "__main__":
    data_directory = "/home/tahsinttalha/MLcode/cats-vs-dogs-classification/data/"
    explore_dataset(data_directory)