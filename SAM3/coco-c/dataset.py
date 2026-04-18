import argparse
import numpy as np
import cv2
import os
import multiprocessing
import concurrent.futures
from tqdm import tqdm

try:
    from imagecorruptions import corrupt, get_corruption_names
except ImportError:
    print("[Error] 'imagecorruptions' is not installed. Please run: pip install imagecorruptions")
    exit()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate COCO-C Dataset (Multiprocessing)")
    parser.add_argument("--data_dir", type=str, default="../../data/coco/val2017", help="Path to original COCO validation images directory")
    parser.add_argument("--save_dir", type=str, default="../../data/coco-c", help="Directory to save the generated COCO-C dataset")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores to use")
    return parser.parse_args()

def process_single_image(img_name, data_dir, save_dir, corruptions_list, severities):
    img_path = os.path.join(data_dir, img_name)
    
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return f"[Warning] Could not read {img_name}"
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    for corruption in corruptions_list:
        for sev in severities:
            corrupted_img_rgb = corrupt(img_rgb, corruption_name=corruption, severity=sev)
            corrupted_img_bgr = cv2.cvtColor(corrupted_img_rgb, cv2.COLOR_RGB2BGR)
            
            save_path = os.path.join(save_dir, corruption, str(sev), img_name)
            cv2.imwrite(save_path, corrupted_img_bgr)
            
    return None

def main(args):
    if not os.path.exists(args.data_dir):
        print(f"[Error] Dataset directory not found at {args.data_dir}")
        return

    valid_exts = ('.jpg', '.jpeg', '.png')
    img_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith(valid_exts)]
    
    if not img_files:
        print(f"[Error] No images found in {args.data_dir}")
        return

    corruptions_list = get_corruption_names()
    severities = [1, 2, 3, 4, 5]

    print(f"[Info] Found {len(img_files)} images in {args.data_dir}")
    print(f"[Info] Using {args.workers} CPU cores for parallel processing...")

    print("[Info] Creating directory structures...")
    for corruption in corruptions_list:
        for sev in severities:
            os.makedirs(os.path.join(args.save_dir, corruption, str(sev)), exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_image, img_name, args.data_dir, args.save_dir, corruptions_list, severities): img_name
            for img_name in img_files
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(img_files), desc="Processing COCO-C (Parallel)"):
            result = future.result()
            if result is not None:
                print(result)

    print("\n[Success] COCO-C dataset generation completed!")

if __name__ == "__main__":
    args = parse_args()
    main(args)