import os
import random
import math
import types
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
    attn_resized = cv2.resize(attn_2d, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    
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
    
    # [핵심 수정 1] SDPA 연산을 끄고 Eager 모드로 로드하여 Attention Map 계산을 강제합니다.
    model = Sam3Model.from_pretrained("facebook/sam3", attn_implementation="eager").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    # ==========================================
    # 2. COCO 데이터셋에서 무작위 이미지 및 Bbox 로드
    # ==========================================
    data_dir = "../../data/coco"
    ann_file = os.path.join(data_dir, "annotations/instances_val2017.json")
    img_dir = os.path.join(data_dir, "val2017")
    for i in range(3):
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
        # 3. Hook 대신 Native Attention 가로채기 (Monkey Patch)
        # ==========================================
        attn_data = {}
        target_layer = model.geometry_encoder.layers[0].cross_attn

        # 원래 레이어의 forward 함수 백업
        original_forward = target_layer.forward

        def hooked_forward(self, *args, **kwargs):
            # 모듈 내부에서 완성된 Attention 확률값을 뱉어내도록 kwargs 주입
            kwargs['output_attentions'] = True
            
            # 원래 동작 수행
            out = original_forward(*args, **kwargs)
            
            # Hugging Face의 Eager Attention은 보통 (hidden_states, attn_weights, ...) 튜플을 반환
            if isinstance(out, tuple) and len(out) > 1:
                # 튜플의 두 번째 값이 우리가 원하는 "진짜" Attention Map
                attn_data['weights'] = out[1].detach().cpu()
                
                # [핵심 수정 부분] 원본 코드가 'hidden_states, _ =' 형태로 2개의 값을 기다리므로, 
                # 모델이 고장나지 않도록 가짜 튜플 형태로 감싸서 반환합니다.
                return (out[0], None)
                
            return out

        # 타겟 레이어의 작동 방식을 새 함수로 덮어씌움
        target_layer.forward = types.MethodType(hooked_forward, target_layer)

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

        # 추출이 끝났으니 원래 모델 상태로 복구 (안전장치)
        target_layer.forward = original_forward

        # ==========================================
    # 5. Attention 형태 분석 및 시각화 (프롬프트 분리)
    # ==========================================
    if 'weights' in attn_data:
        attn_probs = attn_data['weights']
        print(f"[Info] 추출된 Attention 형태: {attn_probs.shape}") 
        # 예상 형태: (1, 8, 2, 5184) -> (Batch, Heads, BBox Points, Image Tokens)

        # Batch 차원 제거 -> (8, 2, 5184)
        attn_probs = attn_probs[0]

        # 8개의 헤드(시야) 중 가장 강하게 반응한 신호만 추출 -> (2, 5184)
        attn_max_heads = attn_probs.max(dim=0).values
        
        # [핵심] 2개의 BBox 프롬프트를 각각 분리합니다.
        attn_top_left = attn_max_heads[0].numpy()     # Q[0]: 좌상단(Top-Left) 포인트
        attn_bottom_right = attn_max_heads[1].numpy() # Q[1]: 우하단(Bottom-Right) 포인트
        
        save_dir = "./analysis/attention_maps"
        os.makedirs(save_dir, exist_ok=True)
        
        base_name = img_info['file_name'].split('.')[0]
        
        # 1. 좌상단 프롬프트가 쳐다보는 곳 시각화
        save_path_1 = os.path.join(save_dir, f"{base_name}_attn_top_left.png")
        visualize_attention_overlay(pil_image, attn_top_left, box_xyxy, save_path=save_path_1)
        
        # 2. 우하단 프롬프트가 쳐다보는 곳 시각화
        save_path_2 = os.path.join(save_dir, f"{base_name}_attn_bottom_right.png")
        visualize_attention_overlay(pil_image, attn_bottom_right, box_xyxy, save_path=save_path_2)
        
        print("[Success] 2개의 프롬프트별 Attention 맵이 각각 분리되어 저장되었습니다!")
    else:
        print("[Error] Attention 데이터를 추출하지 못했습니다.")

if __name__ == "__main__":
    main()