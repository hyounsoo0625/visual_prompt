import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pycocotools.coco import COCO
from transformers import Sam3Processor, Sam3Model
from dotenv import load_dotenv

# HF 토큰 로드 (필요시)
load_dotenv()
HF_CODE = os.environ.get('HF_TOKEN')
if HF_CODE:
    os.environ["HF_TOKEN"] = HF_CODE

def parse_args():
    parser = argparse.ArgumentParser(description="SAM 3 Frequency Domain Analysis on COCO-O")
    
    # 사용자 지정 데이터셋 경로 구조
    parser.add_argument("--ood_base_dir", type=str, default="../../data/ood_coco")
    parser.add_argument("--domain", type=str, default="tattoo", 
                        choices=["cartoon", "handmake", "painting", "sketch", "tattoo", "weather"],
                        help="분석할 OOD 도메인 선택")
    parser.add_argument("--target_class", type=str, default="dog", help="분석할 객체 카테고리")
    
    parser.add_argument("--save_dir", type=str, default="./analysis")
    parser.add_argument("--cutoff_radius", type=int, default=40, help="FFT 저주파/고주파 분할 반경")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def create_frequency_images(pil_img, radius=40):
    """2D FFT를 이용하여 이미지를 저주파(형태)와 고주파(텍스처/노이즈)로 분리합니다."""
    img_np = np.array(pil_img)
    h, w, c = img_np.shape
    
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask_area = (x - cx)**2 + (y - cy)**2 <= radius**2

    lf_img = np.zeros_like(img_np, dtype=np.float32)
    hf_img = np.zeros_like(img_np, dtype=np.float32)

    for i in range(c):
        f_transform = np.fft.fft2(img_np[:, :, i])
        f_shift = np.fft.fftshift(f_transform)

        # 저주파 통과 (Low-pass)
        f_shift_lf = f_shift.copy()
        f_shift_lf[~mask_area] = 0 
        lf_img[:, :, i] = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift_lf)))

        # 고주파 통과 (High-pass)
        f_shift_hf = f_shift.copy()
        f_shift_hf[mask_area] = 0 
        hf_img[:, :, i] = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift_hf)))

    lf_img = np.clip(lf_img, 0, 255).astype(np.uint8)
    hf_img = np.clip(hf_img, 0, 255).astype(np.uint8)

    return Image.fromarray(lf_img), Image.fromarray(hf_img)

def get_sam3_embedding(model, processor, image, box_xyxy, device):
    """SAM 3 모델의 Geometry Encoder에 Hook을 걸어 프롬프트 임베딩을 추출합니다."""
    hooked_embeddings = {}

    def get_geometry_embeds_hook(module, input, output):
        if hasattr(output, 'last_hidden_state'):
            tensor = output.last_hidden_state
        elif hasattr(output, '__getitem__'):
            tensor = output[0]
        else:
            tensor = output
        hooked_embeddings['geometry_out'] = tensor.detach().cpu()

    hook_handle = model.geometry_encoder.register_forward_hook(get_geometry_embeds_hook)
    
    inputs = processor(
        images=image,
        input_boxes=[[box_xyxy]], 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    hook_handle.remove()

    if 'geometry_out' in hooked_embeddings:
        emb = hooked_embeddings['geometry_out']
        v_vec = emb.view(-1)
        # L2 정규화
        v_vec = F.normalize(v_vec, dim=0, p=2)
        return v_vec
    return None

def main(args):
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 데이터셋 경로 설정
    img_dir = os.path.join(args.ood_base_dir, args.domain, "val2017")
    ann_file = os.path.join(args.ood_base_dir, args.domain, "annotations", "instances_val2017.json")
    
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"어노테이션 파일을 찾을 수 없습니다: {ann_file}")
    
    print(f"[Info] Loading COCO annotations for domain: {args.domain}")
    coco = COCO(ann_file)
    
    # 특정 클래스의 어노테이션 가져오기
    cat_ids = coco.getCatIds(catNms=[args.target_class])
    if not cat_ids:
        raise ValueError(f"클래스 '{args.target_class}'를 찾을 수 없습니다.")
        
    ann_ids = coco.getAnnIds(catIds=cat_ids)
    if not ann_ids:
        raise ValueError(f"해당 클래스의 객체가 데이터셋에 존재하지 않습니다.")

    # 랜덤 샘플 하나 선택
    selected_ann_id = random.choice(ann_ids)
    ann = coco.loadAnns(selected_ann_id)[0]
    img_info = coco.loadImgs(ann['image_id'])[0]
    
    img_path = os.path.join(img_dir, img_info['file_name'])
    pil_image = Image.open(img_path).convert("RGB")
    
    # COCO BBox (x, y, w, h) -> SAM BBox (x1, y1, x2, y2) 변환
    x, y, w, h = ann['bbox']
    box_xyxy = [float(x), float(y), float(x + w), float(y + h)]

    print(f"[Info] Selected Image: {img_info['file_name']}, Box: {box_xyxy}")

    # 2. SAM 3 모델 로드
    print(f"[Info] Loading SAM 3 model on {device}...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()

    # 3. 주파수 이미지 분리
    print("[Info] 분할 중: 저주파(형태) 및 고주파(텍스처) 이미지 생성...")
    lf_image, hf_image = create_frequency_images(pil_image, radius=args.cutoff_radius)

    # 4. 임베딩 추출
    print("[Info] 각 도메인별 SAM 3 임베딩 추출 중...")
    emb_orig = get_sam3_embedding(model, processor, pil_image, box_xyxy, device)
    emb_lf = get_sam3_embedding(model, processor, lf_image, box_xyxy, device)
    emb_hf = get_sam3_embedding(model, processor, hf_image, box_xyxy, device)

    if emb_orig is None or emb_lf is None or emb_hf is None:
        print("[Error] 임베딩 추출에 실패했습니다.")
        return

    # 5. 민감도(유사도) 계산
    sim_lf = F.cosine_similarity(emb_orig, emb_lf, dim=0).item()
    sim_hf = F.cosine_similarity(emb_orig, emb_hf, dim=0).item()
    
    total_sim = sim_lf + sim_hf
    reliance_lf = (sim_lf / total_sim) * 100
    reliance_hf = (sim_hf / total_sim) * 100

    print(f"[Result] 형태(Low-Freq) 의존도: {reliance_lf:.2f}%")
    print(f"[Result] 노이즈/텍스처(High-Freq) 의존도: {reliance_hf:.2f}%")

    # 6. 결과 시각화 (논문 Figure용)
    os.makedirs(args.save_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])

    # [Top Row] 원본, 저주파, 고주파 이미지 출력
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(pil_image)
    ax0.set_title(f"1. Original ({args.domain.capitalize()})\nTarget: {args.target_class}", fontweight='bold')
    rect = plt.Rectangle((box_xyxy[0], box_xyxy[1]), box_xyxy[2]-box_xyxy[0], box_xyxy[3]-box_xyxy[1], 
                         fill=False, color='red', linewidth=2)
    ax0.add_patch(rect)
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(lf_image)
    ax1.set_title("2. Low-Frequency (LF)\n(Shape / Structure)", fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(hf_image)
    ax2.set_title("3. High-Frequency (HF)\n(Texture / Noise)", fontweight='bold')
    ax2.axis('off')

    # [Bottom Row] 민감도 그래프 출력
    ax3 = fig.add_subplot(gs[1, :])
    categories = ['Low-Frequency Reliance (Shape)', 'High-Frequency Reliance (Texture/Noise)']
    values = [reliance_lf, reliance_hf]
    colors = ['#3498db', '#e74c3c']

    bars = ax3.barh(categories, values, color=colors, alpha=0.8, height=0.5)
    ax3.set_xlim(0, 100)
    ax3.set_xlabel("Relative Embedding Sensitivity (%)", fontsize=14, fontweight='bold')
    ax3.set_title(f"Prompt Embedding Frequency Analysis (Domain: {args.domain})", fontsize=16, fontweight='bold')

    for bar in bars:
        width = bar.get_width()
        ax3.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                 va='center', ha='left', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(args.save_dir, f"sam3_freq_analysis_{args.domain}_{args.target_class}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Success] Figure saved to {save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)