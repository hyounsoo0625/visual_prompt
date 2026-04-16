import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from pycocotools.coco import COCO
from sklearn.manifold import TSNE
from tqdm import tqdm

# ==========================================
# 1. 초기 설정 (디바이스, 모델, 데이터 경로)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{device}] 환경에서 모델을 로드합니다...")

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

dataDir = '../data/coco'
dataType = 'val2017'
annFile = f'{dataDir}/annotations/instances_{dataType}.json'
imgDir = f'{dataDir}/{dataType}'

# pycocotools 초기화
print("COCO Annotation 로드 중...")
coco = COCO(annFile)

# ==========================================
# 2. Hook 설정 (임베딩 추출용)
# ==========================================
hooked_embeddings = {}

def hook_fn(module, input, output):
    if hasattr(output, '__getitem__') and not isinstance(output, torch.Tensor):
        main_tensor = output[0]
    else:
        main_tensor = output
    hooked_embeddings['geometry_out'] = main_tensor.detach().cpu()

# Hook 등록
hook_handle = model.geometry_encoder.register_forward_hook(hook_fn)

# ==========================================
# 3. 샘플링 및 임베딩 추출
# ==========================================
target_categories = [
    'person', 'car', 'dog', 'cat'
]
samples_per_cat = 20

all_embeddings = []
all_labels = []

# 결과 재현성을 위한 시드 고정 (선택 사항)
random.seed(42)

print("데이터 샘플링 및 임베딩 추출을 시작합니다...")
for cat_name in tqdm(target_categories, desc="Categories"):
    catIds = coco.getCatIds(catNms=[cat_name])
    if not catIds:
        continue
        
    imgIds = coco.getImgIds(catIds=catIds)
    random.shuffle(imgIds) # 이미지를 랜덤하게 섞음
    
    count = 0
    for imgId in imgIds:
        if count >= samples_per_cat:
            break
            
        # 해당 이미지에서 현재 카테고리의 Annotation만 가져옴
        annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        # 이미지 메타데이터 로드
        img_info = coco.loadImgs(imgId)[0]
        img_path = os.path.join(imgDir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        for ann in anns:
            if count >= samples_per_cat:
                break
                
            coco_bbox = ann['bbox'] # [x, y, w, h]
            
            # 너비나 높이가 0 이하인 비정상 박스 패스
            if coco_bbox[2] <= 0 or coco_bbox[3] <= 0:
                continue
                
            # SAM3용 포맷 [x1, y1, x2, y2]으로 변환
            box_xyxy = [
                coco_bbox[0],
                coco_bbox[1],
                coco_bbox[0] + coco_bbox[2],
                coco_bbox[1] + coco_bbox[3]
            ]
            
            # 전처리 및 모델 추론
            inputs = processor(
                images=image,
                input_boxes=[[box_xyxy]],
                input_boxes_labels=[[1]],
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # 임베딩 추출 및 평탄화(Flatten)
            emb = hooked_embeddings['geometry_out']
            emb_flat = emb.view(-1).numpy()
            
            all_embeddings.append(emb_flat)
            all_labels.append(cat_name)
            
            count += 1

# Hook 해제 (메모리 누수 방지)
hook_handle.remove()
print(f"추출 완료: 총 {len(all_embeddings)}개의 임베딩 수집됨.")

# ==========================================
# 4. t-SNE 차원 축소 및 시각화
# ==========================================
print("t-SNE 분석 및 시각화를 진행합니다...")
X = np.array(all_embeddings)

# 데이터가 50개(10 x 5)이므로 perplexity를 낮게 설정해야 함
perplexity_value = min(15, len(X) - 1) 
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
X_tsne = tsne.fit_transform(X)

# Seaborn을 이용한 시각화
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=X_tsne[:, 0], 
    y=X_tsne[:, 1],
    hue=all_labels,
    palette=sns.color_palette("tab10", len(target_categories)), # 10개 카테고리용 색상
    s=150,       # 점 크기 (데이터가 적으므로 크게)
    alpha=0.8,   # 투명도
    edgecolor='w', # 점 테두리
    linewidth=1
)

plt.title("t-SNE of SAM 3 Visual Prompt Embeddings (10 Categories x 5 Samples)", fontsize=16)
plt.xlabel("t-SNE Dim 1", fontsize=12)
plt.ylabel("t-SNE Dim 2", fontsize=12)

# 범례를 그래프 바깥쪽으로 배치
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Categories", fontsize=11, title_fontsize=13)
plt.tight_layout()

# 저장 및 출력
output_filename = "sam3_tsne_10cats_5samples.png"
plt.savefig(output_filename, dpi=300)
plt.show()

print(f"시각화 완료! 결과가 '{output_filename}'로 저장되었습니다.")