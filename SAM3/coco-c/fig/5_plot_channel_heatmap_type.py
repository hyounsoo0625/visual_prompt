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
with open("../analysis/sam3_cococ_features.pkl", "rb") as f:
    data = pickle.load(f)

# ==========================================
# 💡 수정 포인트 1: 특정 손상(brightness)에 대해 Severity 1~5를 순회
# ==========================================
target_corruption = 'zoom_blur'
severities = ['1', '2', '3', '4', '5'] 

channel_stds = []

for sev in severities:
    diffs = []
    for ann_id, v in data.items():
        # 해당 이미지에 clean 피처가 있고, target_corruption의 특정 severity가 존재하는지 확인
        if v['clean'] is not None and target_corruption in v['corrupted'] and sev in v['corrupted'][target_corruption] and v['corrupted'][target_corruption][sev] is not None:
            # 안전한 연산을 위해 1D 배열로 평탄화(flatten) 보장
            clean_emb = v['clean'].flatten()
            corr_emb = v['corrupted'][target_corruption][sev].flatten()
            
            # Clean과 Corrupted의 채널별 절대적 차이 계산
            diff = np.abs(clean_emb - corr_emb)
            diffs.append(diff)
    
    if diffs:
        # 해당 Severity에 대한 채널별 평균 변동폭 계산
        channel_stds.append(np.mean(diffs, axis=0))

heatmap_data = np.array(channel_stds)

# ==========================================
# 💡 수정 포인트 2: 이미지 비율 및 디자인 요소 맞춤
# ==========================================
# 원본 이미지처럼 가로로 길고 세로가 짧은 형태의 비율로 설정
fig, ax = plt.subplots(figsize=(24, 6))

# 💡 여기에 vmin=0.0, vmax=1.5 를 추가하여 컬러바 범위를 고정합니다.
sns.heatmap(heatmap_data, cmap="YlOrRd", vmin=0.0, vmax=1.5, cbar_kws={'label': 'Mean Absolute Deviation'}, xticklabels=12, ax=ax)

# Y축 레이블 변경 (1 ~ 5)
ax.set_yticklabels(severities, rotation=0, fontsize=12)
ax.set_ylabel('Severity Level')

# X축 레이블 및 눈금 각도 설정 (45도 기울임)
ax.set_xlabel('Embedding Dimension / Channels')
plt.xticks(rotation=45)

# 제목 수정 (굵은 글씨로 설정)
ax.set_title(f"Channel Sensitivity Map for '{target_corruption}'", pad=15, weight='bold')

plt.tight_layout()

# 저장 파일명
save_path = f"../analysis/sam3_fig2_channel_heatmap_{target_corruption}.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved {save_path}")