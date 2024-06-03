
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO
import requests

# 모델 파일 다운로드
model_url = 'https://github.com/sf01363/garbage_classification/raw/main/model.pth'
response = requests.get(model_url)
model_path = 'model.pth'
with open(model_path, 'wb') as f:
    f.write(response.content)

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()

        # 사전 학습된 EfficientNet 모델 로드
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        # 기존 분류기 레이어의 특징 수 가져오기
        num_features = self.efficientnet._fc.in_features

        # 커스텀 분류기 정의
        self.efficientnet._fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

model = torch.load(model_path, map_location="cpu")

# 모델의 상태 사전 업데이트
model.eval()

# 클래스 레이블 딕셔너리
class_labels = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
}

st.title('[5조] 재활용 쓰레기 이미지 분류 모형')

file = st.file_uploader('이미지를 올려주세요', type=['jpg', 'png'])

if file is None:
    st.text('이미지를 먼저 올려주세요.')
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # 이미지 전처리
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized).transpose((2, 0, 1))  # HWC to CHW
    img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0  # 정규화

    # 모델 예측
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)

    # 예측 결과 처리
    pred_class = class_labels[predicted.item()]
    st.success(f'Prediction: {pred_class}')
