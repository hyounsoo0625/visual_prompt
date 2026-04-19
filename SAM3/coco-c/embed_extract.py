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

from sam import SAM3 

load_dotenv()
HF_CODE = os.environ.get('HF_TOKEN')
if HF_CODE:
    os.environ["HF_TOKEN"] = HF_CODE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/sam3") 
    parser.add_argument("--data_dir", type=str, default="../../data/coco")
    parser.add_argument("--num_samples", type=int, default=1000, help="평가할 샘플 수")
    parser.add_argument("--save_interval", type=int, default=50, help="중간 저장 간격") 
    return parser.parse_args()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Loading SAM 3 model on {device}...")
    
    base_model = Sam3Model.from_pretrained(args.model).to(device)
    base_processor = Sam3Processor.from_pretrained(args.model)
    sam_extractor = SAM3(model=base_model, processor=base_processor)

    ann_file = os.path.join(args.data_dir, "annotations", "instances_val2017.json")
    coco = COCO(ann_file)
    ann_ids = coco.getAnnIds()[:args.num_samples] 

    corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
    severities = ['1', '2', '3', '4', '5']
    
    os.makedirs("./analysis", exist_ok=True)
    save_path = "./analysis/sam3_cococ_features.pkl"

    # ==========================================
    # 💡 이어하기(Resume) 로직 추가
    # ==========================================
    if os.path.exists(save_path):
        print(f"[Info] 기존 저장 파일 발견: {save_path}")
        with open(save_path, "rb") as f:
            results = pickle.load(f)
        print(f"[Info] {len(results)}개의 데이터가 이미 처리되어 있습니다. 이어서 시작합니다.")
    else:
        results = {}
        print("[Info] 기존 저장 파일이 없습니다. 처음부터 시작합니다.")

    # 아직 처리되지 않은 ann_id만 추려내기
    remaining_ann_ids = [aid for aid in ann_ids if aid not in results]
    print(f"[Info] 전체 대상: {len(ann_ids)}개 | 남은 대상: {len(remaining_ann_ids)}개")

    if len(remaining_ann_ids) == 0:
        print("[Info] 모든 샘플이 이미 처리되었습니다. 스크립트를 종료합니다.")
        return

    print("[Info] Extracting Features...")
    # tqdm은 남은 항목(remaining_ann_ids)에 대해서만 진행
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
                emb = sam_extractor.get_geometry_embeddings(image=pil_image, box_xyxy=box_xyxy)
                if emb is not None:
                    results[ann_id]['clean'] = emb.view(-1).numpy()
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
                    emb = sam_extractor.get_geometry_embeddings(image=pil_image, box_xyxy=box_xyxy)
                    if emb is not None:
                        results[ann_id]['corrupted'][corr][sev] = emb.view(-1).numpy()
                except Exception as e:
                    print(f"[Error] Failed on corrupted image {ann_id}: {e}")

        # ==========================================
        # 💡 핵심: 주기적으로 단일 파일에 덮어쓰기 저장
        # ==========================================
        if (i + 1) % args.save_interval == 0:
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"\n[Checkpoint] {len(results)}/{len(ann_ids)} - Current progress saved to {save_path}")

    sam_extractor.remove_hook()

    # 루프가 무사히 끝났을 때 최종 저장
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
        
    print(f"[Success] All features saved to {save_path}")

if __name__ == "__main__":
    main(parse_args())