import argparse
import numpy as np
import os
import pickle
import torch
from pycocotools.coco import COCO 
from PIL import Image
from transformers import SamModel, SamProcessor 
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
HF_CODE = os.environ.get('HF_TOKEN')
if HF_CODE:
    os.environ["HF_TOKEN"] = HF_CODE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/sam-vit-base") 
    parser.add_argument("--data_dir", type=str, default="../../data/coco")
    parser.add_argument("--num_samples", type=int, default=5000, help="평가할 샘플 수")
    parser.add_argument("--save_interval", type=int, default=50, help="중간 저장 간격") 
    return parser.parse_args()

def extract_decoder_tokens(model, processor, image, box_xyxy, device):
    # 💡 수정 1: 대괄호 3개에서 2개로 변경! [[[box_xyxy]]] -> [[box_xyxy]]
    inputs = processor(images=image, input_boxes=[[box_xyxy]], return_tensors="pt").to(device)
    captured_tokens = []
    
    def hook_fn(module, input, output):
        captured_tokens.append(output[0].detach().cpu())

    hook_handle = model.mask_decoder.transformer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(**inputs)
        
    hook_handle.remove()
    
    # 박스가 정상 입력되었으므로 tokens 형태는 [1, 7, 256]이 됩니다.
    tokens = captured_tokens[0]
    
    # 💡 수정 2: 가장 안전하게 뒤에서 2개(Box 토큰)만 가져오기
    box_tokens = tokens[:, -2:, :] 
    
    # 2개의 토큰(512 차원)을 1D 배열로 평탄화하여 반환
    return box_tokens.view(-1).numpy()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Loading SAM model ({args.model}) on {device}...")
    
    # 공식 SAM 모델 로드
    base_model = SamModel.from_pretrained(args.model).to(device)
    base_processor = SamProcessor.from_pretrained(args.model)

    ann_file = os.path.join(args.data_dir, "annotations", "instances_val2017.json")
    coco = COCO(ann_file)
    ann_ids = coco.getAnnIds()[:args.num_samples] 

    corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
    severities = ['1', '2', '3', '4', '5']
    
    os.makedirs("./analysis", exist_ok=True)
    # 💡 파일명을 _decoder_features.pkl 로 명확히 변경
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

    print("[Info] Extracting Decoder Tokens...")
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
                # 💡 수정: box_xyxy 인자 추가 완료!
                emb = extract_decoder_tokens(base_model, base_processor, pil_image, box_xyxy, device)
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
                    # 💡 수정: extract_image_embeddings -> extract_decoder_tokens로 변경 및 box_xyxy 인자 추가!
                    emb = extract_decoder_tokens(base_model, base_processor, pil_image, box_xyxy, device)
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