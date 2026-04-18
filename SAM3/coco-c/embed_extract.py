import argparse
import numpy as np
import os
import pickle
import torch
from pycocotools.coco import COCO 
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from tqdm import tqdm
from dotenv import load_dotenv

# 새로 분리한 SAM3 클래스를 import (파일명이 sam_utils.py인 경우)
from sam_utils import SAM3 

# HF 토큰 로드
load_dotenv()
HF_CODE = os.environ.get('HF_TOKEN')
if HF_CODE:
    os.environ["HF_TOKEN"] = HF_CODE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/sam3") 
    parser.add_argument("--data_dir", type=str, default="../../data/coco")
    parser.add_argument("--num_samples", type=int, default=500, help="평가할 샘플 수")
    return parser.parse_args()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Loading SAM 3 model on {device}...")
    
    # 1. HuggingFace 원본 모델과 프로세서 로드
    base_model = Sam3Model.from_pretrained(args.model).to(device)
    base_processor = Sam3Processor.from_pretrained(args.model)

    # 2. 커스텀 SAM3 클래스 인스턴스화 (내부적으로 Hook이 자동 등록됨)
    sam_extractor = SAM3(model=base_model, processor=base_processor)

    ann_file = os.path.join(args.data_dir, "annotations", "instances_val2017.json")
    coco = COCO(ann_file)
    ann_ids = coco.getAnnIds()[:args.num_samples] # 빠른 테스트를 위해 샘플 수 제한

    corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
    severities = ['1', '2', '3', '4', '5']
    
    # 데이터 저장용 딕셔너리
    results = {}

    print("[Info] Extracting Features...")
    for ann_id in tqdm(ann_ids):
        ann = coco.loadAnns(ann_id)[0]
        img_info = coco.loadImgs(ann['image_id'])[0]
        file_name = img_info['file_name']
        x, y, w, h = ann['bbox']
        
        if w <= 0 or h <= 0: continue
        
        category_id = ann['category_id']
        box_xyxy = [float(x), float(y), float(x + w), float(y + h)]

        results[ann_id] = {'category_id': category_id, 'clean': None, 'corrupted': {}}
        
        # ==========================================
        # 3. Clean 이미지 특징 추출
        # ==========================================
        clean_path = os.path.join(args.data_dir, "val2017", file_name)
        if os.path.exists(clean_path):
            try:
                pil_image = Image.open(clean_path).convert("RGB")
                
                # 클래스 메서드를 통해 간단히 임베딩 추출
                emb = sam_extractor.get_geometry_embeddings(image=pil_image, box_xyxy=box_xyxy)
                
                if emb is not None:
                    # 1D 배열로 저장
                    results[ann_id]['clean'] = emb.view(-1).numpy()
            except Exception as e:
                pass

        # ==========================================
        # 4. Corrupted (COCO-C) 이미지 특징 추출
        # ==========================================
        for corr in corruptions:
            results[ann_id]['corrupted'][corr] = {}
            for sev in severities:
                corr_path = os.path.join(args.data_dir, "coco-c", corr, sev, file_name)
                if not os.path.exists(corr_path): continue
                
                try:
                    pil_image = Image.open(corr_path).convert("RGB")
                    
                    # 클래스 메서드를 통해 간단히 임베딩 추출
                    emb = sam_extractor.get_geometry_embeddings(image=pil_image, box_xyxy=box_xyxy)
                    
                    if emb is not None:
                        # 1D 배열로 저장
                        results[ann_id]['corrupted'][corr][sev] = emb.view(-1).numpy()
                        
                    # 주의: 원본 코드의 exit(0)는 루프를 완전히 종료시키므로 제거했습니다.
                    # print(results[ann_id])
                    # exit(0)
                    
                except Exception as e:
                    pass

    # 5. 메모리 정리를 위해 Hook 수동 제거
    sam_extractor.remove_hook()

    # 결과 저장
    os.makedirs("./analysis", exist_ok=True)
    save_path = "./analysis/sam3_cococ_features.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
        
    print(f"[Success] Features saved to {save_path}")

if __name__ == "__main__":
    main(parse_args())