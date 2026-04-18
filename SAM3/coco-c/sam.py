import torch

class SAM3:
    def __init__(self, model, processor):
        """
        :param model: HuggingFace Sam3Model 인스턴스
        :param processor: HuggingFace Sam3Processor 인스턴스
        """
        self.model = model
        self.processor = processor
        self.hooked_embeddings = {}
        
        # 1. 초기화 시점에 자동으로 Hook 등록
        self.hook_handle = self.model.geometry_encoder.register_forward_hook(
            self._get_geometry_embeds_hook
        )

    def _get_geometry_embeds_hook(self, module, input, output):
        # 내부에서만 사용하는 메서드이므로 앞에 _(언더스코어)를 붙이는 것이 관례에 맞습니다.
        if hasattr(output, 'last_hidden_state'):
            tensor = output.last_hidden_state
        elif hasattr(output, '__getitem__'):
            tensor = output[0]
        else:
            tensor = output
            
        # GPU 메모리 누수 방지
        self.hooked_embeddings['geometry_out'] = tensor.detach().cpu()

    def get_geometry_embeddings(self, image, box_xyxy=None):
        """
        이미지와 Box 좌표를 받아 geometry embedding을 반환합니다.
        :param image: PIL Image
        :param box_xyxy: [x1, y1, x2, y2] 형태의 Bounding Box 리스트
        """
        # 2. 이전 추론 결과 초기화 (매우 중요)
        self.hooked_embeddings.clear()
        
        # 3. Box 좌표가 있을 때와 없을 때를 구분하여 입력값 생성
        if box_xyxy is not None:
            inputs = self.processor(
                images=image, 
                input_boxes=[[box_xyxy]], # 원본 코드와 동일하게 3중 리스트 유지
                return_tensors="pt"
            ).to(self.model.device)
        else:
            inputs = self.processor(
                images=image, 
                return_tensors="pt"
            ).to(self.model.device)

        with torch.no_grad():
            _ = self.model(**inputs)
            
        return self.hooked_embeddings.get('geometry_out', None)

    def remove_hook(self):
        """
        4. 사용이 끝난 후 메모리 관리를 위해 훅을 수동으로 제거합니다.
        """
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()
            
    def __del__(self):
        """객체가 소멸될 때 자동으로 훅을 제거합니다."""
        self.remove_hook()