import os
import random
import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from transformers import Sam3Processor, Sam3Model
from dotenv import load_dotenv

# ==========================================
# 1. 환경 설정 및 초기화
# ==========================================
load_dotenv()
HF_CODE = os.environ.get('HF_TOKEN')
if HF_CODE:
    os.environ["HF_TOKEN"] = HF_CODE
    print("[Info] HF_TOKEN 로드 완료")
else:
    print("[Warning] HF_TOKEN이 설정되지 않았습니다.")

def visualize_attention_overlay(image_pil, attn_weights, bbox, save_path="cross_attention_overlay.png"):
    """
    1D Attention Weights를 2D 공간으로 변환하여 Bounding Box와 함께 오버레이합니다.
    """
    img_np = np.array(image_pil)
    h_img, w_img = img_np.shape[:2]
    
    num_tokens = attn_weights.shape[0]
    feat_size = int(math.sqrt(num_tokens))
    
    if feat_size * feat_size != num_tokens:
        print(f"[Warning] 토큰 길이가 제곱수가 아닙니다 ({num_tokens}). 시각화를 건너뜁니다.")
        return
        
    attn_2d = attn_weights.reshape(feat_size, feat_size)
    
    # 0 ~ 255 사이로 정규화 (노이즈 억제를 위해 약간의 대비 조정 가능)
    attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
    attn_2d = (attn_2d * 255).astype(np.uint8)
    
    # 원본 이미지 해상도에 맞게 부드럽게 보간(Interpolation)
    attn_resized = cv2.resize(attn_2d, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    
    # 제트(Jet) 컬러맵 적용
    heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 원본 이미지와 6:4 비율로 합성
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    # Bounding Box 그리기 (초록색, 두께 3)
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # 상자 위쪽에 라벨 텍스트 추가 (선택 사항)
    cv2.putText(overlay, "Prompt BBox", (x1, max(y1 - 10, 0)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 결과 시각화 및 저장
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("SAM 3 Cross-Attention Map with BBox", fontsize=16, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    print(f"[Success] Attention 맵이 저장되었습니다: {save_path}")
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] {device} 환경에서 SAM 3 모델을 로드합니다...")
    
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    
    # ==========================================
    # 2. COCO 데이터셋에서 무작위 이미지 및 Bbox 로드
    # ==========================================
    data_dir = "../../data/coco"
    ann_file = os.path.join(data_dir, "annotations/instances_val2017.json")
    img_dir = os.path.join(data_dir, "val2017")
    
    coco = COCO(ann_file)
    valid_ann_ids = coco.getAnnIds()
    random_ann_id = random.choice(valid_ann_ids)
    ann = coco.loadAnns(random_ann_id)[0]
    
    img_info = coco.loadImgs(ann['image_id'])[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    
    if not os.path.exists(img_path):
        print(f"[Error] 이미지를 찾을 수 없습니다: {img_path}")
        return
        
    pil_image = Image.open(img_path).convert("RGB")
    x, y, w, h = ann['bbox']
    box_xyxy = [float(x), float(y), float(x + w), float(y + h)]
    
    print(f"\n[Info] 선택된 이미지: {img_info['file_name']}")
    print(f"[Info] 사용된 프롬프트(Bbox): {box_xyxy}")

    # ==========================================
    # 3. Hook 설정 (Q, K 추출)
    # ==========================================
    attn_data = {}

    def get_q_hook(module, inp, out):
        attn_data['q'] = out.detach().cpu()

    def get_k_hook(module, inp, out):
        attn_data['k'] = out.detach().cpu()

    # Geometry Encoder의 마지막 레이어(인덱스 2) 타겟팅
    target_layer = model.geometry_encoder.layers[2].cross_attn
    h1 = target_layer.q_proj.register_forward_hook(get_q_hook)
    h2 = target_layer.k_proj.register_forward_hook(get_k_hook)

    # ==========================================
    # 4. SAM 3 추론
    # ==========================================
    inputs = processor(
        images=pil_image,
        input_boxes=[[box_xyxy]], 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    h1.remove()
    h2.remove()

    # ==========================================
    # 5. Attention 형태 분석 및 계산
    # ==========================================
    if 'q' in attn_data and 'k' in attn_data:
        q = attn_data['q']
        k = attn_data['k']
        
        hidden_dim = q.shape[-1]
        num_heads = 8 
        head_dim = hidden_dim // num_heads
        
        B, SeqQ, _ = q.shape
        _, SeqK, _ = k.shape
        
        Q = q.view(B, SeqQ, num_heads, head_dim).transpose(1, 2)
        K = k.view(B, SeqK, num_heads, head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # 공간(이미지) 차원을 자동으로 찾기 위한 로직
        # 통상적으로 완벽한 제곱수(예: 64x64=4096)를 갖는 축이 이미지 토큰입니다.
        if math.isqrt(SeqK)**2 == SeqK and SeqK > 10:
            # K가 이미지 토큰인 경우
            attn_mean = attn_probs.mean(dim=(1, 2))[0] # Head와 Query 차원 평균
        elif math.isqrt(SeqQ)**2 == SeqQ and SeqQ > 10:
            # Q가 이미지 토큰인 경우 (각 이미지 픽셀이 프롬프트를 얼마나 참조하는지)
            attn_mean = attn_probs.sum(dim=-1).mean(dim=1)[0]
        else:
            # 기본값 (Fallback)
            attn_mean = attn_probs.mean(dim=(1, 2))[0]
            
        attn_weights_np = attn_mean.numpy()
        
        save_dir = "./analysis/attention_maps"
        save_path = os.path.join(save_dir, f"{img_info['file_name'].split('.')[0]}_attention_with_bbox.png")
        
        # Bbox 정보와 함께 함수 호출
        visualize_attention_overlay(pil_image, attn_weights_np, box_xyxy, save_path=save_path)
    else:
        print("[Error] Attention 데이터를 추출하지 못했습니다.")

if __name__ == "__main__":
    main()