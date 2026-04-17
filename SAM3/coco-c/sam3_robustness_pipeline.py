import os
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from transformers import Sam3Processor, Sam3Model
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# HF 토큰 로드
load_dotenv()
HF_CODE = os.environ.get('HF_TOKEN')
if HF_CODE:
    os.environ["HF_TOKEN"] = HF_CODE
# MacOS 등에서 OMP 에러 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 15가지 학계 표준 Corruption 종류 (제공해주신 폴더 구조와 동일)
CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 
    'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 
    'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur'
]
SEVERITIES = [1, 2, 3, 4, 5]

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
    
    inputs = processor(images=image, input_boxes=[[box_xyxy]], return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)

    hook_handle.remove()

    if 'geometry_out' in hooked_embeddings:
        emb = hooked_embeddings['geometry_out']
        v_vec = emb.view(-1)
        v_vec = F.normalize(v_vec, dim=0, p=2) # L2 정규화
        return v_vec.numpy()
    return None

def extract_features_from_disk(coco_img_dir, coco_c_dir, coco_ann_file, save_path, num_samples=500):
    """디스크에 저장된 COCO-C 이미지를 직접 불러와 임베딩을 추출하고 pkl로 저장합니다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Step 1] Loading SAM 3 model on {device}...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()

    coco = COCO(coco_ann_file)
    
    # 전체 어노테이션 ID 가져오기 및 셔플
    all_ann_ids = coco.getAnnIds()
    random.seed(42)
    random.shuffle(all_ann_ids)
    
    features_dict = {}
    processed_count = 0

    print(f"\n[Step 1] 무작위 {num_samples}개 완전한 객체 임베딩 추출 시작...")
    pbar = tqdm(total=num_samples, desc="Processing Disk Images")
    
    for ann_id in all_ann_ids:
        if processed_count >= num_samples:
            break
            
        ann = coco.loadAnns(ann_id)[0]
        img_info = coco.loadImgs(ann['image_id'])[0]
        img_name = img_info['file_name']
        
        # BBox 검증 (너무 작은 박스 필터링)
        x, y, w, h = ann['bbox']
        if w <= 10 or h <= 10: 
            continue
            
        box_xyxy = [float(x), float(y), float(x + w), float(y + h)]
        
        # 1. Clean 이미지 존재 여부 확인
        clean_img_path = os.path.join(coco_img_dir, img_name)
        if not os.path.exists(clean_img_path): 
            continue

        # 2. 15가지 Corruption x 5가지 Severity 이미지가 모두 존재하는지 사전 검증
        # (하나라도 파일이 없으면 이 객체는 과감히 포기하고 다음 객체로 넘어감)
        is_complete = True
        for corr in CORRUPTIONS:
            for sev in SEVERITIES:
                corrupted_path = os.path.join(coco_c_dir, corr, str(sev), img_name)
                if not os.path.exists(corrupted_path):
                    is_complete = False
                    break
            if not is_complete:
                break
        
        if not is_complete:
            continue  # 누락된 파일이 있으므로 continue

        # --- 여기서부터는 모든 파일이 온전히 존재함이 보장됨 ---
        try:
            # Clean 임베딩 추출
            clean_pil = Image.open(clean_img_path).convert("RGB")
            clean_emb = get_sam3_embedding(model, processor, clean_pil, box_xyxy, device)
            if clean_emb is None: 
                continue
            
            obj_data = {'clean': clean_emb, 'corrupted': {corr: {} for corr in CORRUPTIONS}}
            extraction_failed = False

            # Corrupted 임베딩 추출 (디스크에서 로드)
            for corr in CORRUPTIONS:
                for sev in SEVERITIES:
                    corrupted_path = os.path.join(coco_c_dir, corr, str(sev), img_name)
                    corr_pil = Image.open(corrupted_path).convert("RGB")
                    corr_emb = get_sam3_embedding(model, processor, corr_pil, box_xyxy, device)
                    
                    if corr_emb is None:
                        extraction_failed = True
                        break
                    obj_data['corrupted'][corr][str(sev)] = corr_emb
                
                if extraction_failed:
                    break
            
            # 모델 내부 에러 등으로 하나라도 실패했다면 저장하지 않음
            if extraction_failed:
                continue

            # 모든 과정이 성공했을 때만 딕셔너리에 추가
            features_dict[ann_id] = obj_data
            processed_count += 1
            pbar.update(1)

        except Exception as e:
            # 이미지 파일이 깨져서 안 열리는 등의 에러 발생 시 무시하고 넘어감
            pass

    pbar.close()

    # pkl 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(features_dict, f)
    print(f"[Step 1 완료] 총 {processed_count}개의 완벽한 특징 벡터 묶음 저장 완료: {save_path}")

def plot_figure_1_severity_drop(pkl_path, save_path):
    print("\n[Step 2] Figure 1 (Severity Drop) 시각화 시작...")
    plt.rcParams.update({'font.family': 'serif', 'axes.labelsize': 14, 'axes.titlesize': 16, 'legend.fontsize': 12})
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', 
              '#fabebe', '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000']
    str_severities = [str(s) for s in SEVERITIES]

    fig, ax = plt.subplots(figsize=(18, 12))

    for idx, corr in enumerate(CORRUPTIONS):
        means, stds = [], []
        for sev in str_severities:
            sims = []
            for ann_id, v in data.items():
                if v['clean'] is not None and sev in v['corrupted'][corr] and v['corrupted'][corr][sev] is not None:
                    clean_emb = v['clean'].reshape(1, -1)
                    corr_emb = v['corrupted'][corr][sev].reshape(1, -1)
                    sim = cosine_similarity(clean_emb, corr_emb)[0][0]
                    sims.append(sim)
            
            means.append(np.mean(sims) if sims else 0)
            stds.append(np.std(sims) if sims else 0)
            
        ax.errorbar(str_severities, means, yerr=stds, label=corr.replace('_', ' ').title(), 
                    color=colors[idx], marker='o', capsize=5, linewidth=2, markersize=8)

    ax.set_xlabel('Corruption Severity')
    ax.set_ylabel('Cosine Similarity to Clean Prototype')
    ax.set_title('SAM 3 Feature Robustness across 500 Disk Samples (COCO-C)')
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Step 2 완료] Saved: {save_path}")
def plot_figure_2_channel_heatmap(pkl_path, save_path):
    print("\n[Step 3] Figure 2 (Channel Sensitivity Heatmap) 시각화 시작...")
    plt.rcParams.update({'font.family': 'serif', 'axes.labelsize': 14, 'axes.titlesize': 16})
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    severity = '5'
    channel_stds = []

    for corr in CORRUPTIONS:
        diffs = []
        for ann_id, v in data.items():
            if v['clean'] is not None and severity in v['corrupted'][corr] and v['corrupted'][corr][severity] is not None:
                clean_emb = v['clean'].flatten()
                corr_emb = v['corrupted'][corr][severity].flatten()
                diff = np.abs(clean_emb - corr_emb)
                diffs.append(diff)
        
        if diffs:
            channel_stds.append(np.mean(diffs, axis=0))

    heatmap_data = np.array(channel_stds)
    num_channels = heatmap_data.shape[1] # 전체 512 채널

    # 가로로 긴 Figure 생성
    fig, ax = plt.subplots(figsize=(40, 8))
    
    # 💡 수정 포인트: xticklabels=10 을 추가하여 10 간격으로만 채널 인덱스를 출력하도록 Seaborn에 위임
    sns.heatmap(heatmap_data, cmap="YlOrRd", cbar_kws={'label': 'Mean Absolute Deviation'}, 
                ax=ax, xticklabels=10)

    # y축 라벨 설정
    ax.set_yticklabels([c.replace('_', ' ').title() for c in CORRUPTIONS], rotation=0, fontsize=12)
    
    # 💡 수정 포인트: set_xticklabels 대신 tick_params를 사용하여 폰트 크기와 회전만 제어
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    
    ax.set_xlabel('Embedding Dimension / Channel Index (All Channels)')
    ax.set_title(f'SAM 3 Channel Sensitivity Map (Severity 5, All {num_channels} Channels, 500 Disk Samples)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Step 3 완료] Saved: {save_path}")

if __name__ == "__main__":
    # --- 데이터 경로 설정 (사용자 환경에 맞게 조정) ---
    COCO_IMG_DIR = "../../data/coco/val2017"
    COCO_C_DIR = "../../data/coco-c"  # 손상된 이미지가 있는 루트 디렉토리
    COCO_ANN_FILE = "../../data/coco/annotations/instances_val2017.json"
    
    PKL_SAVE_PATH = "./analysis/sam3_disk_500_features.pkl"
    FIG1_SAVE_PATH = "./analysis/sam3_fig1_disk_500_severity_drop.png"
    FIG2_SAVE_PATH = "./analysis/sam3_fig2_disk_500_channel_heatmap.png"

    # Step 1: 디스크에서 읽어와 특징 추출 (500개)
    # extract_features_from_disk(COCO_IMG_DIR, COCO_C_DIR, COCO_ANN_FILE, PKL_SAVE_PATH, num_samples=5)

    # Step 2 & 3: 시각화
    plot_figure_1_severity_drop(PKL_SAVE_PATH, FIG1_SAVE_PATH)
    plot_figure_2_channel_heatmap(PKL_SAVE_PATH, FIG2_SAVE_PATH)
    
    print("\n✅ 전체 랜덤 500개 샘플(Disk Load)에 대한 분석 및 시각화가 완료되었습니다!")