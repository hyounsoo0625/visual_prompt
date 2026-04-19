import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image
from transformers import SamModel, SamProcessor # SAM 3 사용 시 Sam3Model, Sam3Processor로 변경
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="COCO-O Heatmap Analysis")
    # 경로 설정
    parser.add_argument("--data_dir", type=str, default="../../data", help="ood_coco 폴더가 있는 상위 경로")
    parser.add_argument("--db_path", type=str, default="./analysis/sam3_embedding_db.pkl")
    parser.add_argument("--save_path", type=str, default="./analysis/cocoo_category_heatmap.png")
    
    # DB 생성을 위한 설정
    parser.add_argument("--target_domains", type=str, nargs='+', 
                        default=["cartoon", "handmake", "painting", "sketch", "tattoo", "weather"])
    parser.add_argument("--target_categories", type=str, nargs='+', 
                        default=["person", "dog", "car", "bird", "cat", "boat", "bottle"])
    parser.add_argument("--sample_per_domain", type=int, default=200, help="도메인당 추출할 최대 객체 수")
    
    # 모델 설정
    parser.add_argument("--model_id", type=str, default="facebook/sam-vit-base")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()

def get_embedding(model, processor, img_path, bbox, device):
    """이미지 경로와 bbox를 받아 임베딩을 추출 (SAM Mask Decoder 토큰 기준)"""
    try:
        image = Image.open(img_path).convert("RGB")
        x, y, w, h = bbox
        box_xyxy = [float(x), float(y), float(x + w), float(y + h)]
        
        inputs = processor(images=image, input_boxes=[[box_xyxy]], return_tensors="pt").to(device)
        
        captured = []
        def hook(m, i, o): captured.append(o[0].detach().cpu())
        handle = model.mask_decoder.transformer.register_forward_hook(hook)
        
        with torch.no_grad():
            _ = model(**inputs)
        handle.remove()
        
        # 마지막 2개 토큰(Box 관련) 사용 (평탄화 후 정규화)
        tokens = captured[0][:, -2:, :].view(-1)
        return F.normalize(tokens, dim=0).numpy()
    except Exception as e:
        return None

def main(args):
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    
    # ==========================================
    # 1. DB 파일이 없으면 생성
    # ==========================================
    if not os.path.exists(args.db_path):
        print(f"[Info] DB 파일이 없습니다. COCO-O에서 임베딩을 추출합니다...")
        model = SamModel.from_pretrained(args.model_id).to(args.device)
        processor = SamProcessor.from_pretrained(args.model_id)
        
        database = []
        for dom in args.target_domains:
            ann_file = os.path.join(args.data_dir, "ood_coco", dom, "annotations", "instances_val2017.json")
            img_dir = os.path.join(args.data_dir, "ood_coco", dom, "val2017")
            
            if not os.path.exists(ann_file): continue
            coco = COCO(ann_file)
            cat_ids = coco.getCatIds(catNms=args.target_categories)
            ann_ids = coco.getAnnIds(catIds=cat_ids)
            
            # 도메인당 샘플 제한
            if len(ann_ids) > args.sample_per_domain:
                ann_ids = np.random.choice(ann_ids, args.sample_per_domain, replace=False)
            
            print(f" -> Processing Domain: {dom} ({len(ann_ids)} objects)")
            for aid in tqdm(ann_ids):
                ann = coco.loadAnns(int(aid))[0]
                img_info = coco.loadImgs(ann['image_id'])[0]
                cat_name = coco.loadCats(ann['category_id'])[0]['name']
                
                emb = get_embedding(model, processor, os.path.join(img_dir, img_info['file_name']), ann['bbox'], args.device)
                if emb is not None:
                    database.append({
                        'domain': dom,
                        'category_name': cat_name,
                        'embedding': emb
                    })
        
        with open(args.db_path, 'wb') as f:
            pickle.dump(database, f)
        print(f"[Success] DB created with {len(database)} embeddings.")
        # 모델 메모리 해제
        del model
        torch.cuda.empty_cache()
    
    # ==========================================
    # 2. 히트맵 시각화 (사용자 요청 방식)
    # ==========================================
    print(f"[Info] Loading DB for Heatmap: {args.db_path}")
    with open(args.db_path, 'rb') as f:
        database = pickle.load(f)

    # 카테고리별 그룹화
    category_embeddings = {cat: [] for cat in args.target_categories}
    for item in database:
        cat_name = item.get('category_name')
        if cat_name in category_embeddings:
            category_embeddings[cat_name].append(item['embedding'])

    heatmap_data = []
    y_labels = []

    for cat in args.target_categories:
        embs = category_embeddings[cat]
        if len(embs) < 2: continue
        
        # 분산 계산 (Mean Absolute Deviation으로 변경 시 np.mean(np.abs(...)) 사용)
        variances = np.var(np.vstack(embs), axis=0)
        heatmap_data.append(variances)
        y_labels.append(cat.capitalize())

    if not heatmap_data:
        print("[Error] No data to plot."); return

    # 그리기
    heatmap_matrix = np.array(heatmap_data)
    fig, ax = plt.subplots(figsize=(18, len(y_labels) * 0.8))
    cax = ax.imshow(heatmap_matrix, aspect='auto', cmap='YlOrRd')

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')
    ax.set_xlabel("Embedding Channel Index", fontsize=12)
    ax.set_title(f"SAM Channel Variance Map (COCO-O Domains Combined)", fontsize=15, pad=20)

    cbar = fig.colorbar(cax, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Variance Across Domains", fontsize=11)

    for spine in ax.spines.values(): spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    print(f"[Success] Heatmap saved to {args.save_path}")

if __name__ == "__main__":
    main(parse_args())