import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO 
from transformers import Sam3Processor, Sam3Model
from PIL import Image
from sklearn.manifold import TSNE
import random
from tqdm import tqdm
from dotenv import load_dotenv

# HF 토큰 로드
load_dotenv()
HF_CODE = os.environ.get('HF_TOKEN')
if HF_CODE:
    os.environ["HF_TOKEN"] = HF_CODE
    print("[Info] HF_TOKEN 로드 완료")

def parse_args():
    parser = argparse.ArgumentParser(description="SAM 3 Clean vs OOD t-SNE Analysis")
    
    parser.add_argument("--coco_img_dir", type=str, default="../../data/coco/val2017")
    parser.add_argument("--coco_ann_file", type=str, default="../../data/coco/annotations/instances_val2017.json")
    parser.add_argument("--coco_o_base_dir", type=str, default="../../data/ood_coco") 
    parser.add_argument("--ood_ann_filename", type=str, default="instances_val2017.json") 
    
    parser.add_argument("--save_dir", type=str, default="./analysis")
    parser.add_argument("--samples_per_class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Loading SAM 3 model on {device}...")
    
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    # ==========================================
    # 1. Hook 설정 (Geometry Encoder)
    # ==========================================
    hooked_embeddings = {}

    def get_geometry_embeds_hook(module, input, output):
        if hasattr(output, 'last_hidden_state'):
            tensor = output.last_hidden_state
        elif hasattr(output, '__getitem__'):
            tensor = output[0]
        else:
            tensor = output
        # GPU 메모리 누수 방지
        hooked_embeddings['geometry_out'] = tensor.detach().cpu()

    hook_handle = model.geometry_encoder.register_forward_hook(get_geometry_embeds_hook)
    print("[Info] Geometry Encoder Hook attached.")

    random.seed(args.seed)
    
    target_classes = ['person', 'bicycle', 'car', 'cat', 'dog']
    # target_classes = ['person']
    domains = ['clean', 'sketch', 'cartoon', 'weather', 'painting', 'handmake', 'tattoo']
    
    embeddings = []
    labels_class = []
    labels_type = []

    print(f"\n[Info] Starting Data Extraction...")
    
    for domain in domains:
        if domain == 'clean':
            img_dir = args.coco_img_dir
            ann_file = args.coco_ann_file
            data_type = 'Clean' 
        else:
            img_dir = os.path.join(args.coco_o_base_dir, domain, "val2017")
            ann_file = os.path.join(args.coco_o_base_dir, domain, "annotations", args.ood_ann_filename)
            data_type = 'OOD (COCO-O)' 
            
        if not os.path.exists(ann_file): 
            print(f"[Warning] Annotation file not found for domain '{domain}'. Skipping...")
            continue
            
        try:
            coco = COCO(ann_file)
            for cls_name in target_classes:
                cat_ids = coco.getCatIds(catNms=[cls_name])
                if not cat_ids: continue
                
                ann_ids = coco.getAnnIds(catIds=cat_ids)
                if not ann_ids: continue
                
                # 샘플링
                selected_ann_ids = random.sample(ann_ids, min(len(ann_ids), args.samples_per_class))
                
                for ann_id in tqdm(selected_ann_ids, desc=f"  Extracting {cls_name} ({data_type})", leave=False):
                    ann = coco.loadAnns(ann_id)[0]
                    img_info = coco.loadImgs(ann['image_id'])[0]
                    
                    x, y, w, h = ann['bbox']
                    if w <= 0 or h <= 0: continue
                    
                    img_path = os.path.join(img_dir, img_info['file_name'])
                    if not os.path.exists(img_path): continue
                    
                    try:
                        pil_image = Image.open(img_path).convert("RGB")
                    except Exception:
                        continue

                    # Bounding Box 좌표 (SAM 3 형식)
                    x1, y1 = float(x), float(y)
                    x2, y2 = float(x + w), float(y + h)
                    if x2 <= x1 or y2 <= y1: continue
                    
                    box_xyxy = [x1, y1, x2, y2]

                    # ==========================================
                    # 2. SAM 3 추론 및 임베딩 추출
                    # ==========================================
                    hooked_embeddings.clear()
                    
                    inputs = processor(
                        images=pil_image,
                        input_boxes=[[box_xyxy]], # 3중 리스트 유지
                        return_tensors="pt"
                    ).to(device)

                    with torch.no_grad():
                        _ = model(**inputs)

                    if 'geometry_out' in hooked_embeddings:
                        emb = hooked_embeddings['geometry_out']
                        v_vec = emb.view(-1)
                        if v_vec.numel() == 0: continue
                        
                        # L2 정규화
                        v_vec = F.normalize(v_vec, dim=0, p=2)
                        
                        embeddings.append(v_vec.numpy())
                        labels_class.append(cls_name)
                        labels_type.append(data_type)
                        
        except Exception as e:
            print(f"[Error] Error processing domain '{domain}': {e}")
            pass

    # Hook 제거
    hook_handle.remove()

    if len(embeddings) == 0:
        print("[Error] No embeddings were extracted. Check your data paths and token.")
        return

    # ==========================================
    # 3. t-SNE 및 시각화
    # ==========================================
    print(f"\n[Info] Running t-SNE on {len(embeddings)} embeddings...")
    X = np.array(embeddings)
    perplexity_val = min(40, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=args.seed, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    df_tsne = pd.DataFrame({
        't-SNE 1': X_2d[:, 0],
        't-SNE 2': X_2d[:, 1],
        'Class': labels_class,
        'Type': labels_type
    })

    print("[Info] Generating 2-Color t-SNE Plots...")
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # [왼쪽] 클래스별 시각화
    sns.scatterplot(
        data=df_tsne, x='t-SNE 1', y='t-SNE 2', 
        hue='Class', palette='Set1', s=60, alpha=0.8, ax=axes[0]
    )
    axes[0].set_title(f"SAM 3 t-SNE: By Class ({len(target_classes)} Classes)", fontsize=18, fontweight='bold')
    axes[0].legend(title="Class", loc='upper right')

    # [오른쪽] Clean vs OOD 도메인 시각화
    sns.scatterplot(
        data=df_tsne, x='t-SNE 1', y='t-SNE 2', 
        hue='Type', palette=['#3498db', '#e74c3c'], 
        style='Class', markers=['o', 's', 'D', '^', 'v'], 
        s=80, alpha=0.7, ax=axes[1]
    )
    axes[1].set_title("SAM 3 t-SNE: Clean vs OOD (Domain Shift)", fontsize=18, fontweight='bold')
    axes[1].legend(title="Type & Class", loc='upper right', bbox_to_anchor=(1.25, 1))

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout()
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "sam3_tsne_clean_vs_ood.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Success] Plot saved to {save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)