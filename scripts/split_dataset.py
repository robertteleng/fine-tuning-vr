#!/usr/bin/env python3
"""
Split dataset into train/val sets.
"""

import shutil
import random
from pathlib import Path
from tqdm import tqdm


def get_matching_pairs(images_dir, labels_dir):
    """Find image files that have corresponding label files."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    pairs = []

    # Find all images
    for img_path in images_dir.glob('*.jpg'):
        label_path = labels_dir / f"{img_path.stem}.txt"

        # Only include if label exists
        if label_path.exists():
            pairs.append({
                'image': img_path,
                'label': label_path
            })

    return pairs


def split_pairs(pairs, train_ratio=0.8, seed=42):
    """Split pairs into train and val sets."""
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)

    train = shuffled[:split_idx]
    val = shuffled[split_idx:]

    return train, val

def copy_files(pairs, output_dir, split_name):
    """Copy image and label files to output directory."""
    images_out = Path(output_dir) / split_name / 'images'
    labels_out = Path(output_dir) / split_name / 'labels'
    
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    for pair in tqdm(pairs, desc=f"Copying {split_name}"):
        shutil.copy(pair['image'], images_out / pair['image'].name)
        shutil.copy(pair['label'], labels_out / pair['label'].name)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Split dataset into train/val')
    parser.add_argument('--images', type=str, required=True, help='Folder with images')
    parser.add_argument('--labels', type=str, required=True, help='Folder with labels')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--ratio', type=float, default=0.8, help='Train ratio (default 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Find matching pairs
    pairs = get_matching_pairs(args.images, args.labels)
    print(f"Found {len(pairs)} image-label pairs")

    # Split
    train, val = split_pairs(pairs, args.ratio, args.seed)
    print(f"Train: {len(train)}, Val: {len(val)}")

    # Copy files
    copy_files(train, args.output, 'train')
    copy_files(val, args.output, 'val')

    print(f"\nDone! Dataset saved to {args.output}")


if __name__ == '__main__':
    main()