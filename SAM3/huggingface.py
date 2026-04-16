import torch
import json
import os
from PIL import Image
from transformers import Sam3Processor, Sam3Model

# 1. 디바이스 및 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# 2. 파일 경로 설정
image_filename = '000000077595.jpg'
image_path = f'../data/coco/val2017/{image_filename}'
annotation_path = '../data/coco/annotations/instances_val2017.json'

# 이미지 로드
image = Image.open(image_path).convert("RGB")

# 3. COCO Annotation에서 바운딩 박스 추출 및 변환
print("Annotation 파일을 읽는 중...")
with open(annotation_path, 'r') as f:
    coco_data = json.load(f)

# 파일명에서 image_id 추출 (예: '000000077595.jpg' -> 77595)
image_id = int(os.path.splitext(image_filename)[0])

# 해당 image_id를 가진 모든 annotation 찾기
image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

if len(image_annotations) == 0:
    raise ValueError(f"이미지 ID {image_id}에 대한 annotation을 찾을 수 없습니다.")

# 첫 번째 객체의 바운딩 박스 선택 (필요에 따라 인덱스를 변경하여 다른 객체 선택 가능)
coco_bbox = image_annotations[0]['bbox']  # [x, y, width, height]
category_id = image_annotations[0]['category_id']
print(f"찾은 COCO BBox (x, y, w, h): {coco_bbox}, 카테고리 ID: {category_id}")

# COCO [x, y, w, h] 포맷을 SAM3 [x1, y1, x2, y2] 포맷으로 변환
box_xyxy = [
    coco_bbox[0],                      # x_min
    coco_bbox[1],                      # y_min
    coco_bbox[0] + coco_bbox[2],       # x_max = x_min + width
    coco_bbox[1] + coco_bbox[3]        # y_max = y_min + height
]
print(f"SAM 3 입력용 BBox (x1, y1, x2, y2): {box_xyxy}")

# 4. 전처리
inputs = processor(
    images=image,
    input_boxes=[[box_xyxy]],
    input_boxes_labels=[[1]], # 1은 해당 박스를 포함(Positive)하라는 의미
    return_tensors="pt"
).to(device)
# 1. 추출한 임베딩을 저장할 딕셔너리
hooked_embeddings = {}

# 2. Hook 함수 정의
# 2. Hook 함수 정의 수정본
def hook_fn(module, input, output):
    # Hugging Face의 전용 Output 객체이거나 튜플/리스트인 경우
    # 메인 임베딩 텐서는 보통 첫 번째 인덱스[0]에 들어있습니다.
    if hasattr(output, '__getitem__') and not isinstance(output, torch.Tensor):
        main_tensor = output[0]
    else:
        # 혹시라도 단일 텐서로 나올 경우
        main_tensor = output
        
    # 추출한 텐서에 detach() 적용
    hooked_embeddings['geometry_out'] = main_tensor.detach().cpu()

# 3. Hook 등록 (geometry_encoder에 부착)
hook_handle = model.geometry_encoder.register_forward_hook(hook_fn)

# 4. Forward Pass 실행 (여기가 핵심입니다!)
print("모델 추론 및 Hook 추출 중...")
with torch.no_grad():
    # 내부 인코더를 억지로 호출하지 않고, 전체 모델에 inputs를 넘깁니다.
    # 이렇게 하면 모델이 정상적으로 작동하면서 Hook이 자동으로 발동됩니다.
    outputs = model(**inputs)

# 5. Hook 해제 (메모리 누수 방지)
hook_handle.remove()

# 6. 결과 확인
visual_prompt_emb = hooked_embeddings['geometry_out']

print("-" * 30)
print(f"Hook로 추출한 Embedding Shape: {visual_prompt_emb.shape}")
