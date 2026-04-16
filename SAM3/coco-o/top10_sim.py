import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import random
import pickle 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    parser = argparse.ArgumentParser(description="SAM 3 Visual Prompt Top-10 Retrieval (PKL Cache & Multi-Query)")
    parser.add_argument("--device", type=str, default="0", help="사용할 GPU 번호 (예: '0') 또는 'cpu'")
    
    parser.add_argument("--target_domains", type=str, nargs='+', default=["cartoon", "handmake", "painting", "sketch", "tattoo", "weather"])
    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument("--query_domain", type=str, default="cartoon")
    
    parser.add_argument("--num_queries", type=int, default=10, help="테스트할 쿼리의 개수")
    parser.add_argument("--sample_size_per_domain", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./analysis")
    
    parser.add_argument("--db_filename", type=str, default="sam3_embedding_db.pkl", help="캐싱할 DB 파일명")
    return parser.parse_args()

def draw_bbox(ax, img_path, bbox, color='red', title=""):
    img = cv2.imread(img_path)
    if img is None: return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    x, y, w, h = bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[Info] PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and args.device != 'cpu':
        device = f"cuda:{args.device}"
        print(f"[Info] Using GPU device: {device} ({torch.cuda.get_device_name(int(args.device))})")
    else:
        device = "cpu"
        print("[Warning] CUDA is not available or 'cpu' specified. Using CPU. This will be slow!")

    # ==========================================
    # SAM 3 모델 로드 및 Hook 설정
    # ==========================================
    try:
        print(f"[Info] Loading SAM 3 model on {device}...")
        model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3")
    except Exception as e:
        print(f"[Error] Failed to load SAM 3 model: {e}")
        return

    hooked_embeddings = {}
    def get_geometry_embeds_hook(module, input, output):
        if hasattr(output, 'prompt_cross_attn'):
            tensor = output.prompt_cross_attn
        elif hasattr(output, '__getitem__'):
            tensor = output[0]
        else:
            tensor = output
        hooked_embeddings['geometry_out'] = tensor.detach().cpu()

    hook_handle = model.mask_decoder.register_forward_hook(get_geometry_embeds_hook)
    print("[Info] Geometry Encoder Hook attached.")

    def get_embedding(img_path, bbox):
        # 1. 파일이 실제로 존재하는지 먼저 확인
        if not os.path.exists(img_path):
            print(f"[Warning] Image file not found: {img_path}")
            return None

        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Error] Failed to open image {img_path}: {e}")
            return None

        x, y, w, h = bbox
        box_xyxy = [float(x), float(y), float(x + w), float(y + h)]
        
        hooked_embeddings.clear()
        try:
            inputs = processor(
                images=pil_image,
                input_boxes=[[box_xyxy]], # 3중 리스트 
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                _ = model(**inputs)
            
            if 'geometry_out' in hooked_embeddings:
                emb = hooked_embeddings['geometry_out']
                v_vec = emb.view(-1)
                if v_vec.numel() == 0: return None
                
                v_vec = F.normalize(v_vec, dim=0, p=2)
                return v_vec.numpy()
                
        except Exception as e:
            # 2. 에러가 발생하면 무시하지 않고 출력!
            print(f"\n[Error in Model Inference] {e}")
            return None
            
        return None

    # ==========================================
    # 1. Target Database 구축 (PKL 캐싱 적용)
    # ==========================================
    db_path = os.path.join(args.save_dir, args.db_filename)
    database = []

    if os.path.exists(db_path):
        print(f"\n[Info] Found existing embedding DB at {db_path}. Loading data...")
        with open(db_path, 'rb') as f:
            database = pickle.load(f)
        print(f"[Info] Loaded {len(database)} embeddings from PKL.")
    else:
        print(f"\n[Info] No existing DB found. Extracting embeddings to create {db_path}...")
        for dom in args.target_domains:
            target_ann_file = os.path.join(args.data_dir, "ood_coco", dom, "annotations", "instances_val2017.json")
            target_img_dir = os.path.join(args.data_dir, "ood_coco", dom, "val2017")
            
            # 💡 경로를 제대로 찾는지 확인하기 위한 출력문 추가
            if not os.path.exists(target_ann_file):
                print(f"[Warning] Annotation file not found for '{dom}'.")
                print(f" -> Checked path: {os.path.abspath(target_ann_file)}") # 절대 경로로 출력
                continue

            print(f"\n[Info] Processing target domain: {dom}")
            try:
                target_coco = COCO(target_ann_file)
            except Exception as e:
                print(f"[Error] Failed to load COCO file for {dom}: {e}")
                continue

            target_ann_ids = target_coco.getAnnIds()
            if args.sample_size_per_domain > 0 and len(target_ann_ids) > args.sample_size_per_domain:
                target_ann_ids = random.sample(target_ann_ids, args.sample_size_per_domain)

            print(f"       Extracting embeddings for {len(target_ann_ids)} objects...")
            for ann_id in tqdm(target_ann_ids, desc=f"Domain: {dom}"):
                ann = target_coco.loadAnns(ann_id)[0]
                x, y, w, h = ann['bbox']
                if w <= 0 or h <= 0: continue

                img_info = target_coco.loadImgs(ann['image_id'])[0]
                img_path = os.path.join(target_img_dir, img_info['file_name'])
                
                emb = get_embedding(img_path, ann['bbox'])
                if emb is not None:
                    database.append({
                        'domain': dom,
                        'ann_id': ann_id,
                        'img_path': img_path,
                        'bbox': ann['bbox'],
                        'embedding': emb
                    })
        
        if not database:
            print("[Error] No valid embeddings extracted. Exiting.")
            hook_handle.remove()
            return

        with open(db_path, 'wb') as f:
            pickle.dump(database, f)
        print(f"\n[Info] Successfully saved {len(database)} embeddings to {db_path}.")

    # ==========================================
    # 2. Query 선택 및 탐색 루프
    # ==========================================
    query_ann_file = os.path.join(args.data_dir, "ood_coco", args.query_domain, "annotations", "instances_val2017.json")
    query_img_dir = os.path.join(args.data_dir, "ood_coco", args.query_domain, "val2017")

    try:
        query_coco = COCO(query_ann_file)
    except Exception as e:
        print(f"[Error] Failed to load Query COCO annotations: {e}")
        hook_handle.remove()
        return

    all_query_ids = query_coco.getAnnIds()
    valid_query_ids = []
    for q_id in all_query_ids:
        ann = query_coco.loadAnns(q_id)[0]
        if ann['bbox'][2] > 10 and ann['bbox'][3] > 10:
            valid_query_ids.append(q_id)

    if len(valid_query_ids) < args.num_queries:
        print(f"[Warning] Not enough valid queries. Using {len(valid_query_ids)} instead of {args.num_queries}.")
        selected_queries = valid_query_ids
    else:
        selected_queries = random.sample(valid_query_ids, args.num_queries)

    print(f"\n[Info] Starting search for {len(selected_queries)} queries...")

    for idx, query_ann_id in enumerate(selected_queries):
        print(f"\n--- Processing Query {idx+1}/{len(selected_queries)} (Ann ID: {query_ann_id}) ---")
        
        query_ann = query_coco.loadAnns(query_ann_id)[0]
        query_img_info = query_coco.loadImgs(query_ann['image_id'])[0]
        query_img_path = os.path.join(query_img_dir, query_img_info['file_name'])
        query_bbox = query_ann['bbox']

        query_emb = get_embedding(query_img_path, query_bbox)
        if query_emb is None:
            print(f"[Warning] Failed to extract embedding for Query {query_ann_id}. Skipping.")
            continue

        # 3. 거리 계산 (자기 자신 제외)
        results = []
        for item in database:
            if item['domain'] == args.query_domain and item['ann_id'] == query_ann_id:
                continue
            
            dist = np.linalg.norm(query_emb - item['embedding'])
            results.append({**item, 'distance': dist})

        if not results:
            continue

        results_sorted = sorted(results, key=lambda x: x['distance'])
        top_10_results = results_sorted[:10]

        # 4. 시각화 및 개별 파일 저장
        fig = plt.figure(figsize=(20, 13))
        
        ax_query = plt.subplot2grid((3, 5), (0, 2))
        query_title = f"[QUERY]\nDomain: {args.query_domain}\nID: {query_ann_id}"
        draw_bbox(ax_query, query_img_path, query_bbox, color='red', title=query_title)

        for i, result in enumerate(top_10_results):
            row = (i // 5) + 1
            col = i % 5
            ax_res = plt.subplot2grid((3, 5), (row, col))
            title_text = f"Rank {i+1} [{result['domain']}]\nDist: {result['distance']:.3f}"
            draw_bbox(ax_res, result['img_path'], result['bbox'], color='lime', title=title_text)

        plt.tight_layout()

        save_name = f"sam3_query_{idx+1}_id_{query_ann_id}.png"
        final_save_path = os.path.join(args.save_dir, save_name)
        plt.savefig(final_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Success] Saved result to {final_save_path}")

    hook_handle.remove()
    print("\n[Info] All queries processed completely!")

if __name__ == "__main__":
    args = parse_args()
    main(args)