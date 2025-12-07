#!/usr/bin/env python3
"""
Visualize YOLO annotations on images.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def load_annotations(label_path):
    """Load YOLO annotations from .txt file."""
    annotations = []
    
    if not Path(label_path).exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                annotations.append({
                    'class': int(parts[0]),
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'w': float(parts[3]),
                    'h': float(parts[4])
                })
    
    return annotations

def yolo_to_pixels(ann, img_width, img_height):
    """Convert YOLO normalized coords to pixel coords."""
    x_center = ann['x'] * img_width
    y_center = ann['y'] * img_height
    w = ann['w'] * img_width
    h = ann['h'] * img_height
    
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)
    
    return x1, y1, x2, y2


def draw_annotations(image, annotations, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on image."""
    img_height, img_width = image.shape[:2]
    annotated = image.copy()
    
    for ann in annotations:
        x1, y1, x2, y2 = yolo_to_pixels(ann, img_width, img_height)
        
        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Draw class label
        label = f"class {ann['class']}"
        cv2.putText(annotated, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return annotated

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize YOLO annotations')
    parser.add_argument('--images', type=str, required=True, help='Folder with images')
    parser.add_argument('--labels', type=str, required=True, help='Folder with .txt labels')
    parser.add_argument('--output', type=str, required=True, help='Output folder for annotated images')
    parser.add_argument('--sample', type=int, default=None, help='Random sample N images (optional)')
    
    args = parser.parse_args()
    
    # Create output folder
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    images = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    
    # Optional: random sample
    if args.sample and args.sample < len(images):
        images = random.sample(images, args.sample)
        print(f"Sampled {args.sample} random images")
    
    print(f"Processing {len(images)} images...")
    
    # Process each image
    total_annotations = 0
    for img_path in tqdm(images):
        # Load image (color for visualization)
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Find corresponding label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        annotations = load_annotations(label_path)
        
        # Draw and save
        annotated = draw_annotations(image, annotations)
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), annotated)
        
        total_annotations += len(annotations)
    
    print(f"\nDone! {len(images)} images saved to {output_dir}")
    print(f"Total annotations drawn: {total_annotations}")


if __name__ == '__main__':
    main()