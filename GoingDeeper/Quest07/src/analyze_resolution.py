import os
import glob
from PIL import Image
import numpy as np

def analyze_resolution():
    base_dir = os.path.join(os.getenv("HOME"), "work/project/GoingDeeper/Quest07/data")
    image_dir = os.path.join(base_dir, "training", "image_2")

    print(f"Searching for images in: {image_dir}")

    if not os.path.exists(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return

    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_paths:
        print("No images found.")
        return

    print(f"Found {len(image_paths)} images.")

    widths = []
    heights = []
    aspect_ratios = []

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

    avg_width = np.mean(widths)
    avg_height = np.mean(heights)
    avg_aspect_ratio = np.mean(aspect_ratios)

    print("-" * 30)
    print(f"Average Width: {avg_width:.2f}")
    print(f"Average Height: {avg_height:.2f}")
    print(f"Average Aspect Ratio: {avg_aspect_ratio:.2f}")
    print("-" * 30)

    # Suggest resolutions divisible by 32
    print("Suggested Resolutions (divisible by 32):")
    
    # Try to match aspect ratio while keeping dimensions reasonable
    # Target height around 256 or 160 (common for UNet)
    
    target_heights = [160, 224, 256, 320, 384]
    
    for h in target_heights:
        w = int(h * avg_aspect_ratio)
        # Round to nearest multiple of 32
        w = round(w / 32) * 32
        print(f"  {w} x {h} (Aspect Ratio: {w/h:.2f})")

if __name__ == "__main__":
    analyze_resolution()
