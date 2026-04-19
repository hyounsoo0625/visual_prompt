import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 학술 논문용 matplotlib 세팅
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# 💡 SAM 3 모델로 추출한 pkl 파일을 불러오도록 이름 수정
with open("../analysis/sam-vit-base_decoder_features.pkl", "rb") as f:
    data = pickle.load(f)
sample_id = list(data.keys())[0]
print(f"총 이미지 개수: {len(data)}")
print(f"첫 번째 이미지의 Clean 임베딩 유무: {data[sample_id]['clean'] is not None}")
print(f"첫 번째 이미지의 Motion Blur (Sev 5) 데이터: {data[sample_id]['corrupted'].get('motion_blur', {}).get('5')}")
corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
severities = ['1', '2', '3', '4', '5']
colors = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', 
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', 
    '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000'
]

fig, ax = plt.subplots(figsize=(18, 16))

for idx, corr in enumerate(corruptions):
    means, stds = [], []
    for sev in severities:
        sims = []
        for ann_id, v in data.items():
            if v['clean'] is not None and sev in v['corrupted'][corr] and v['corrupted'][corr][sev] is not None:
                # 1D array일 수 있으므로 (1, -1)로 reshape 처리
                clean_emb = v['clean'].reshape(1, -1)
                corr_emb = v['corrupted'][corr][sev].reshape(1, -1)
                sim = cosine_similarity(clean_emb, corr_emb)[0][0]
                sims.append(sim)
        
        means.append(np.mean(sims) if sims else 0)
        stds.append(np.std(sims) if sims else 0)
        
    # 에러바가 포함된 Line Plot
    ax.errorbar(severities, means, yerr=stds, label=corr.replace('_', ' ').title(), 
                color=colors[idx], marker='o', capsize=5, linewidth=2, markersize=8)

ax.set_xlabel('Corruption Severity')
ax.set_ylabel('Cosine Similarity to Clean Prototype')
# 💡 그래프 제목을 SAM 3로 수정
ax.set_title('SAM 3 Feature Robustness against Corruptions')
ax.set_ylim(0.0, 1.05)
ax.grid(True, linestyle='--', alpha=0.6)

# 범례 위치를 조금 더 깔끔하게 조정 (데이터가 가려지지 않도록)
ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=10)

plt.tight_layout()
# 💡 저장되는 이미지 파일 이름도 sam3용으로 수정
save_path = "../analysis/sam_fig1_severity_drop.png"
plt.savefig(save_path, dpi=300)
print(f"Saved: {save_path}")