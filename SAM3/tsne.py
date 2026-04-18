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
from dotenv import load_dotenv

load_dotenv()
HF_CODE = os.environ.get('HF_TOKEN')
print("HF_TOKEN 로드 완료")
os.environ["HF_TOKEN"] = HF_CODE

# ==========================================
# 1. 초기 설정 (디바이스, 모델, 데이터 경로)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{device}] 환경에서 모델을 로드합니다...")

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
print(model)
exit(0)
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

# Forward Hook 함수 정의
def get_geometry_embeds_hook(module, input, output):
    # Hugging Face의 ModelOutput(예: Sam3GeometryEncoderOutput) 처리
    if hasattr(output, 'last_hidden_state'):
        # 속성이 명시적으로 있는 경우
        tensor = output.last_hidden_state
    elif hasattr(output, '__getitem__'):
        # 튜플처럼 인덱싱이 가능한 경우 (첫 번째 요소가 메인 임베딩 텐서)
        tensor = output[0]
    else:
        tensor = output
    
    # GPU에 있는 텐서를 CPU로 옮기고 계산 그래프에서 분리한 뒤 저장
    hooked_embeddings['geometry_out'] = tensor.detach().cpu()

# geometry_encoder의 최종 출력단에 Hook 등록
hook_handle = model.geometry_encoder.register_forward_hook(get_geometry_embeds_hook)
print("Geometry Encoder에 Forward Hook 등록 완료.")


# ==========================================
# 3. 샘플링 및 임베딩 추출
# ==========================================
target_categories = [
    'person', 'car', 'dog', 'cat', 'handbag', 'truck', 'bench'
]
samples_per_cat = 100

all_embeddings = []
all_labels = []

# 결과 재현성을 위한 시드 고정
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
                input_boxes_labels=[[1]], # Positive prompt
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Hook을 통해 추출된 임베딩 가져오기
            emb = hooked_embeddings['geometry_out']
            
            # 1D 배열로 평탄화(Flatten)하여 numpy 형태로 저장
            emb_flat = emb.view(-1).numpy()
            
            all_embeddings.append(emb_flat)
            all_labels.append(cat_name)
            
            count += 1

# Hook 해제 (메모리 누수 방지)
hook_handle.remove()
print(f"\n추출 완료: 총 {len(all_embeddings)}개의 임베딩 수집됨.")

# ==========================================
# 4. t-SNE 차원 축소 및 시각화
# ==========================================
print("t-SNE 분석 및 시각화를 진행합니다...")
X = np.array(all_embeddings)

# 데이터 개수에 맞춰 perplexity 자동 조절
perplexity_value = min(30, len(X) - 1) 
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
X_tsne = tsne.fit_transform(X)

# Seaborn을 이용한 시각화
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=X_tsne[:, 0], 
    y=X_tsne[:, 1],
    hue=all_labels,
    palette=sns.color_palette("tab10", len(target_categories)), # 카테고리 수에 맞춘 색상
    s=150,       # 점 크기
    alpha=0.8,   # 투명도
    edgecolor='w', # 점 테두리
    linewidth=1
)

# 동적 타이틀
num_cats = len(target_categories)
title_str = f"t-SNE of SAM 3 Visual Prompt Embeddings ({num_cats} Categories x {samples_per_cat} Samples)"
plt.title(title_str, fontsize=16)
plt.xlabel("t-SNE Dim 1", fontsize=12)
plt.ylabel("t-SNE Dim 2", fontsize=12)

# 범례를 그래프 바깥쪽으로 배치
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Categories", fontsize=11, title_fontsize=13)
plt.tight_layout()

# 저장 및 출력
output_filename = f"sam3_tsne_{num_cats}cats_{samples_per_cat}samples.png"
plt.savefig(output_filename, dpi=300)
plt.show()

print(f"시각화 완료! 결과가 '{output_filename}'로 저장되었습니다.")