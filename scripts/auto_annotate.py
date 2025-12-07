#!/usr/bin/env python3
"""
Auto-anotación de imágenes usando template matching.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_template(template_path: str) -> np.ndarray:
    """Load template image in grayscale."""
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Could not load template: {template_path}")
    return template

def match_template_multiscale(image, template, scales=None, threshold=0.65):
    """Search for template in image at different scales."""
    if scales is None:
        scales = np.linspace(0.3, 2.0, 20)  # 20 scales from 30% to 200%

    detections = []
    h_template, w_template = template.shape[:2]

    for scale in scales:
        # Resize template
        new_w = int(w_template * scale)
        new_h = int(h_template * scale)
        
        # Skip if too large or too small
        if new_w >= image.shape[1] or new_h >= image.shape[0]:
            continue
        if new_w < 10 or new_h < 10:
            continue
        
        scaled_template = cv2.resize(template, (new_w, new_h))
        
        # Template matching
        result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
        
        # Save where there is a match
        locations = np.where(result >= threshold)
        for (y, x) in zip(*locations):
            detections.append({
                'x': x, 'y': y,
                'w': new_w, 'h': new_h,
                'conf': result[y, x]
            })

    return detections

def nms(detections, iou_threshold=0.3):
    """Remove duplicate detections using Non-Maximum Suppression."""
    if len(detections) == 0:
        return []
    
    # Convert to arrays
    boxes = np.array([[d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h']] 
                      for d in detections])
    scores = np.array([d['conf'] for d in detections])
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by confidence (highest first)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Keep boxes with IoU below threshold
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    
    return [detections[i] for i in keep]

def to_yolo_format(detections, img_width, img_height, class_id=0):
    """Convert detections to YOLO format (normalized)."""
    lines = []
    
    for det in detections:
        # Calculate center
        x_center = (det['x'] + det['w'] / 2) / img_width
        y_center = (det['y'] + det['h'] / 2) / img_height
        
        # Normalize dimensions
        width = det['w'] / img_width
        height = det['h'] / img_height
        
        # Format: class x_center y_center width height
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lines.append(line)
    
    return lines

def process_image(image_path, template, threshold=0.65, iou_threshold=0.3):
    """Process a single image: detect and filter."""
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"  Error: Could not load {image_path}")
        return []
    
    height, width = image.shape[:2]
    
    # Detect
    detections = match_template_multiscale(image, template, threshold=threshold)
    
    # Filter duplicates
    filtered = nms(detections, iou_threshold=iou_threshold)
    
    # Convert to YOLO
    yolo_lines = to_yolo_format(filtered, width, height)
    
    return yolo_lines


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-annotation with template matching')
    parser.add_argument('--frames', type=str, required=True, help='Folder with images')
    parser.add_argument('--template', type=str, required=True, help='Template image')
    parser.add_argument('--output', type=str, required=True, help='Output folder for labels')
    parser.add_argument('--threshold', type=float, default=0.65, help='Detection threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.3, help='NMS IoU threshold (0-1)')
    
    args = parser.parse_args()
    
    # Create output folder
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load template
    print(f"Loading template: {args.template}")
    template = load_template(args.template)
    print(f"  Size: {template.shape[1]}x{template.shape[0]} px")
    
    # Find images
    frames_dir = Path(args.frames)
    images = sorted(list(frames_dir.glob('*.jpg')) + list(frames_dir.glob('*.png')))
    print(f"Images found: {len(images)}")
    
    # Process each image
    total_detections = 0
    for img_path in tqdm(images, desc="Processing"):
        lines = process_image(str(img_path), template, args.threshold, args.iou)
        
        # Save .txt file
        label_path = output_dir / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))
        
        total_detections += len(lines)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Images processed: {len(images)}")
    print(f"Total detections: {total_detections}")
    print(f"Average per image: {total_detections/max(len(images),1):.1f}")
    print(f"Labels saved to: {output_dir}")


if __name__ == '__main__':
    main()