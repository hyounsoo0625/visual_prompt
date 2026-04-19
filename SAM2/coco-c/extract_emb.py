import argparse
import numpy as np
import os
import pickle
import torch
from pycocotools.coco import COCO 
from PIL import Image

# 💡 SAM 2 공식 모델과 프로세서를 임포트합니다.
from transformers import Sam2Model, Sam2Processor 
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
HF_CODE = os.environ.get('HF_TOKEN')
if HF_CODE:
    os.environ["HF_TOKEN"] = HF_CODE

def parse_args():
    parser = argparse.ArgumentParser()
    # 💡 기본 모델을 최신 SAM 2.1 (또는 SAM 2) 모델로 변경했습니다.
    # 사용 목적에 따라 facebook/sam2.1-hiera-large 등으로 변경하실 수 있습니다.
    parser.add_argument("--model", type=str, default="facebook/sam2-hiera-base-plus") 
    parser.add_argument("--data_dir", type=str, default="../../data/coco")
    parser.add_argument("--num_samples", type=int, default=1000, help="평가할 샘플 수")
    parser.add_argument("--save_interval", type=int, default=50, help="중간 저장 간격") 
    return parser.parse_args()

def extract_decoder_tokens_sam2(model, processor, image, box_xyxy, device):
    """
    💡 SAM 2 Mask Decoder 내부의 Transformer에서 
    이미지 정보를 흡수한 프롬프트 토큰을 가로채어 추출하는 함수
    """
    inputs = processor(images=image, input_boxes=[[[box_xyxy]]], return_tensors="pt").to(device)
    
    captured_tokens = []
    
    # 1. 후크(Hook) 함수 정의
    def hook_fn(module, input, output):
        # output[0]은 Transformer를 통과하며 Cross-Attention을 마친 query 토큰들입니다.
        captured_tokens.append(output[0].detach().cpu())

    # 2. SAM 2 Mask Decoder 내부 Transformer에 후크 달기
    # (만약 transformers 버전에 따라 경로가 다르다는 에러가 나면, 
    # print(model.mask_decoder) 를 통해 내부 transformer 모듈 이름을 확인해 맞춰주시면 됩니다)
    hook_handle = model.mask_decoder.transformer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(**inputs)
        
    # 3. 메모리 누수를 막기 위해 후크 제거
    hook_handle.remove()
    
    # 4. 토큰 전처리 및 반환
    tokens = captured_tokens[0]
    
    # ==========================================
    # 💡 핵심 방어 로직: SAM 2의 구조적 변화(메모리 토큰 등)에 대비하여, 
    # 토큰의 총 개수와 무관하게 항상 "마지막 2개(Box 토큰)"만 정확히 슬라이싱합니다.
    # ==========================================
    box_tokens = tokens[:, -2:, :] 
    
    # 2개의 토큰을 1D 배열로 평탄화하여 반환 (유사도 비교용)
    return box_tokens.view(-1).numpy()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Loading SAM 2 model ({args.model}) on {device}...")
    
    # 💡 SAM 2 모델 로드 (Bfloat16 등을 사용하면 메모리를 아낄 수 있습니다)
    base_model = Sam2Model.from_pretrained(args.model).to(device)
    base_processor = Sam2Processor.from_pretrained(args.model)

    ann_file = os.path.join(args.data_dir, "annotations", "instances_val2017.json")
    coco = COCO(ann_file)
    ann_ids = coco.getAnnIds()[:args.num_samples] 

    corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
    severities = ['1', '2', '3', '4', '5']
    
    os.makedirs("./analysis", exist_ok=True)
    # 파일명을 sam2 용으로 분리하여 저장
    save_path = f"./analysis/{args.model.split('/')[-1]}_decoder_features.pkl"

    if os.path.exists(save_path):
        print(f"[Info] 기존 저장 파일 발견: {save_path}")
        with open(save_path, "rb") as f:
            results = pickle.load(f)
        print(f"[Info] {len(results)}개의 데이터가 이미 처리되어 있습니다. 이어서 시작합니다.")
    else:
        results = {}
        print("[Info] 기존 저장 파일이 없습니다. 처음부터 시작합니다.")

    remaining_ann_ids = [aid for aid in ann_ids if aid not in results]
    print(f"[Info] 전체 대상: {len(ann_ids)}개 | 남은 대상: {len(remaining_ann_ids)}개")

    if len(remaining_ann_ids) == 0:
        print("[Info] 모든 샘플이 이미 처리되었습니다. 스크립트를 종료합니다.")
        return

    print("[Info] Extracting SAM 2 Decoder Tokens...")
    for i, ann_id in enumerate(tqdm(remaining_ann_ids)):
        ann = coco.loadAnns(ann_id)[0]
        img_info = coco.loadImgs(ann['image_id'])[0]
        file_name = img_info['file_name']
        x, y, w, h = ann['bbox']
        
        if w <= 0 or h <= 0: continue
        
        category_id = ann['category_id']
        box_xyxy = [float(x), float(y), float(x + w), float(y + h)]

        results[ann_id] = {'category_id': category_id, 'clean': None, 'corrupted': {}}
        
        # 3. Clean 이미지 추출
        clean_path = os.path.join(args.data_dir, "val2017", file_name)
        if os.path.exists(clean_path):
            try:
                pil_image = Image.open(clean_path).convert("RGB")
                emb = extract_decoder_tokens_sam2(base_model, base_processor, pil_image, box_xyxy, device)
                results[ann_id]['clean'] = emb
            except Exception as e:
                print(f"[Error] Failed on clean image {ann_id}: {e}")

        # 4. Corrupted 이미지 추출
        for corr in corruptions:
            results[ann_id]['corrupted'][corr] = {}
            for sev in severities:
                corr_path = os.path.join('../../data', "coco-c", corr, sev, file_name)
                if not os.path.exists(corr_path): continue
                
                try:
                    pil_image = Image.open(corr_path).convert("RGB")
                    emb = extract_decoder_tokens_sam2(base_model, base_processor, pil_image, box_xyxy, device)
                    results[ann_id]['corrupted'][corr][sev] = emb
                except Exception as e:
                    print(f"[Error] Failed on corrupted image {ann_id}: {e}")

        # 중간 저장
        if (i + 1) % args.save_interval == 0:
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"\n[Checkpoint] {len(results)}/{len(ann_ids)} - Current progress saved to {save_path}")

    # 최종 저장
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
        
    print(f"[Success] All features saved to {save_path}")

if __name__ == "__main__":
    main(parse_args())