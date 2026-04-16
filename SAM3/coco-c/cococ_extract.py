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

# HF 토큰 로드
load_dotenv()
HF_CODE = os.environ.get('HF_TOKEN')
if HF_CODE:
    os.environ["HF_TOKEN"] = HF_CODE
def parse_args():
    parser = argparse.ArgumentParser()
    # model 인자는 SAM3를 사용하므로 HF 경로를 기본값으로 변경했습니다.
    parser.add_argument("--model", type=str, default="facebook/sam3") 
    parser.add_argument("--data_dir", type=str, default="../../data/coco")
    parser.add_argument("--num_samples", type=int, default=500, help="평가할 샘플 수")
    return parser.parse_args()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Loading SAM 3 model on {device}...")
    
    # YOLOE 대신 SAM 3 모델 로드
    model = Sam3Model.from_pretrained(args.model).to(device)
    processor = Sam3Processor.from_pretrained(args.model)

    visual_features = {}
    
    # SAM 3 Geometry Encoder Hook 정의
    def get_geometry_embeds_hook(module, input, output):
        if hasattr(output, 'last_hidden_state'):
            tensor = output.last_hidden_state
        elif hasattr(output, '__getitem__'):
            tensor = output[0]
        else:
            tensor = output
        # 메모리 누수 방지
        visual_features['geometry_out'] = tensor.detach().cpu()

    # Hook 등록
    model.detr_decoder.register_forward_hook(get_geometry_embeds_hook)

    ann_file = os.path.join(args.data_dir, "annotations", "instances_val2017.json")
    coco = COCO(ann_file)
    ann_ids = coco.getAnnIds()[:args.num_samples] # 빠른 테스트를 위해 샘플 수 제한

    corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur'] # 논문용 대표 15종
    severities = ['1', '2', '3', '4', '5']
    
    # 데이터 저장용 딕셔너리 구조: results[ann_id][corruption][severity] = embedding
    results = {}

    print("[Info] Extracting Features...")
    for ann_id in tqdm(ann_ids):
        ann = coco.loadAnns(ann_id)[0]
        img_info = coco.loadImgs(ann['image_id'])[0]
        file_name = img_info['file_name']
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0: continue
        
        category_id = ann['category_id']
        
        # SAM 3용 Bounding Box 포맷 (3중 리스트를 위해 기본 좌표만 리스트로 묶어둠)
        box_xyxy = [float(x), float(y), float(x + w), float(y + h)]

        results[ann_id] = {'category_id': category_id, 'clean': None, 'corrupted': {}}
        
        # 1. Clean 이미지 추출 (val2017 원본이 있다고 가정)
        clean_path = os.path.join(args.data_dir, "val2017", file_name)
        if os.path.exists(clean_path):
            visual_features.clear()
            try:
                pil_image = Image.open(clean_path).convert("RGB")
                inputs = processor(
                    images=pil_image,
                    input_boxes=[[box_xyxy]], # SAM3 입력 형태: 3중 리스트
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    _ = model(**inputs)
                
                # 1D 배열로 저장
                results[ann_id]['clean'] = visual_features['geometry_out'].view(-1).numpy()
            except: pass

        # 2. Corrupted 이미지 추출
        for corr in corruptions:
            results[ann_id]['corrupted'][corr] = {}
            for sev in severities:
                corr_path = os.path.join(args.data_dir, "coco-c", corr, sev, file_name)
                if not os.path.exists(corr_path): continue
                
                visual_features.clear()
                try:
                    pil_image = Image.open(corr_path).convert("RGB")
                    inputs = processor(
                        images=pil_image,
                        input_boxes=[[box_xyxy]], 
                        return_tensors="pt"
                    ).to(device)

                    with torch.no_grad():
                        _ = model(**inputs)
                    
                    # 1D 배열로 저장
                    results[ann_id]['corrupted'][corr][sev] = visual_features['geometry_out'].view(-1).numpy()
                    print(results[ann_id])
                    exit(0)
                except: pass

    # 결과 저장
    os.makedirs("./analysis", exist_ok=True)
    with open("./analysis/sam3_cococ_features.pkl", "wb") as f: # 파일명만 sam3용으로 살짝 변경
        pickle.dump(results, f)
    print("[Success] Features saved to ./analysis/sam3_cococ_features.pkl")

if __name__ == "__main__":
    main(parse_args())