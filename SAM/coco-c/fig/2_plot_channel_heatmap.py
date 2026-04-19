import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 학술 논문용 matplotlib 세팅
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# SAM 3 특징 파일 로드
with open("../analysis/sam-vit-base_decoder_features.pkl", "rb") as f:
    data = pickle.load(f)

corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
severity = '5' # 가장 극한의 상황(5)에서 채널 민감도 비교

channel_stds = []

for corr in corruptions:
    diffs = []
    for ann_id, v in data.items():
        if v['clean'] is not None and severity in v['corrupted'][corr] and v['corrupted'][corr][severity] is not None:
            # 안전한 연산을 위해 1D 배열로 평탄화(flatten) 보장
            clean_emb = v['clean'].flatten()
            corr_emb = v['corrupted'][corr][severity].flatten()
            
            # Clean과 Corrupted의 채널별 절대적 차이 계산
            diff = np.abs(clean_emb - corr_emb)
            diffs.append(diff)
    
    if diffs:
        # 해당 손상에 대한 채널별 평균 변동폭
        channel_stds.append(np.mean(diffs, axis=0))

heatmap_data = np.array(channel_stds)

# ==========================================
# 💡 수정 포인트: 상위 50개 슬라이싱 제거, 전체 데이터 사용
# ==========================================
# 전체 채널을 시각화하므로, 가로 크기를 넓게 유지
fig, ax = plt.subplots(figsize=(24, 10))

# 데이터 적용 (전체 heatmap_data)
# xticklabels=50 옵션으로 x축 눈금이 너무 빽빽해지는 것을 방지 (50 채널마다 표시)
sns.heatmap(heatmap_data, cmap="YlOrRd", cbar_kws={'label': 'Mean Absolute Deviation'}, xticklabels=50, ax=ax)

ax.set_yticklabels([c.replace('_', ' ').title() for c in corruptions], rotation=0, fontsize=12)

# x축 레이블 변경 (Top 50 문구 제거)
ax.set_xlabel('Embedding Dimension / Channel Index')

# 제목 수정
ax.set_title('SAM 3 Channel Sensitivity Map (All Channels, Severity 5)')

plt.tight_layout()

# 저장 파일명 (선택적으로 변경 가능)
save_path = "../analysis/sam_fig2_channel_heatmap_all.png"
plt.savefig(save_path, dpi=300)
print(f"Saved {save_path}")