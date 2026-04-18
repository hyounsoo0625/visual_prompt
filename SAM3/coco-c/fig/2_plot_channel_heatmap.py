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
with open("./analysis/sam3_cococ_features.pkl", "rb") as f:
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
# 💡 수정 포인트: 실제로 가장 민감한 상위 50개 채널 추출
# ==========================================
# 전체 손상(Corruption)에 대해 평균적으로 가장 크게 변동한 채널 순위 계산
overall_mean_diff = np.mean(heatmap_data, axis=0)
top_50_indices = np.argsort(overall_mean_diff)[::-1][:50] # 내림차순 정렬 후 상위 50개

# 상위 50개 채널만 슬라이싱
heatmap_data_top50 = heatmap_data[:, top_50_indices]

# 가로 크기를 50개 채널 비율에 맞게 적절히 조정 (기존 32는 너무 넓을 수 있음)
fig, ax = plt.subplots(figsize=(24, 10))

# 데이터 적용 (heatmap_data_top50)
sns.heatmap(heatmap_data_top50, cmap="YlOrRd", cbar_kws={'label': 'Mean Absolute Deviation'}, ax=ax)

ax.set_yticklabels([c.replace('_', ' ').title() for c in corruptions], rotation=0, fontsize=12)

# x축 레이블에 '실제 채널 인덱스 번호'를 표기하여 논문 분석 시 특정 채널을 지목할 수 있게 도움
ax.set_xticklabels(top_50_indices, rotation=90, fontsize=9) 
ax.set_xlabel('Embedding Dimension / Channel Index (Top 50 Most Sensitive)')

# 💡 제목 수정 (SAM 3 명시)
ax.set_title('SAM 3 Channel Sensitivity Map (Severity 5)')

plt.tight_layout()

# 💡 저장 파일명 수정
save_path = "./analysis/sam3_fig2_channel_heatmap.png"
plt.savefig(save_path, dpi=300)
print(f"Saved {save_path}")