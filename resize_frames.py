"""
One-time script: resize all clip frames from 1920x1080 -> 224x224 in-place.
Run ONCE before training. Makes each epoch ~3-4x faster.

Usage (from E:\\Dynamic Facial Expression folder):
    python resize_frames.py
"""

import os
import glob
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

CLIP_ROOT = r"E:\Dynamic Facial Expression\Clip\clip_224x224"
TARGET_SIZE = (224, 224)
NUM_THREADS = 8  # adjust to your CPU core count

lock = threading.Lock()
counter = [0]
skipped = [0]


def resize_clip(clip_dir):
    jpg_files = glob.glob(os.path.join(clip_dir, "*.jpg"))
    for jpg_path in jpg_files:
        try:
            img = Image.open(jpg_path)
            # Skip if already 224x224 or smaller
            if img.size[0] <= 224 and img.size[1] <= 224:
                with lock:
                    skipped[0] += 1
                img.close()
                continue
            img_resized = img.resize(TARGET_SIZE, Image.BILINEAR)
            img.close()
            img_resized.save(jpg_path, "JPEG", quality=95)
            img_resized.close()
        except Exception as e:
            print(f"  ERROR on {jpg_path}: {e}")

    with lock:
        counter[0] += 1
        if counter[0] % 500 == 0:
            print(f"  Processed {counter[0]} clips... ({skipped[0]} frames already small)")


def main():
    clip_dirs = [
        os.path.join(CLIP_ROOT, d)
        for d in os.listdir(CLIP_ROOT)
        if os.path.isdir(os.path.join(CLIP_ROOT, d))
    ]

    total = len(clip_dirs)
    print(f"Found {total} clip folders. Resizing to {TARGET_SIZE} using {NUM_THREADS} threads...")
    print("This runs ONCE and saves in-place. Grab a coffee.")

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(resize_clip, d): d for d in clip_dirs}
        for future in as_completed(futures):
            future.result()  # raise any exceptions

    print(f"\nDone! Processed {counter[0]} clips. {skipped[0]} frames were already small.")
    print("You can now run training -- epochs will be ~3-4x faster.")


if __name__ == "__main__":
    main()
