
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO
import requests
from torchvision import transforms


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

device = 'cpu'
model = torch.load(model_path, map_location="cpu")

# 모델의 상태 사전 업데이트
model.eval()

# 클래스 레이블 딕셔너리
class_labels = {
    0: '골판지',
    1: '유리',
    2: '금속',
    3: '종이',
    4: '플라스틱',
}

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title('재활용 쓰레기 이미지 분류')

file = st.file_uploader('이미지를 올려주세요', type=['jpg', 'png'])

if file is None:
    st.text('이미지를 먼저 올려주세요.')
else:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # 이미지 전처리
    image = transform(image).unsqueeze(0).to(device)
    
    # 모델 예측
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    # 예측 확률 슬라이더 추가
    threshold = st.slider('예측 확률 임계값을 선택하세요', 0.0, 1.0, 0.5, 0.01)

    # 예측 결과 처리
    pred_class = class_labels[predicted.item()]
    pred_probability = probabilities[0, predicted.item()].item()
    
    if pred_probability >= threshold:
        st.success(f'{pred_class}일 확률은 {pred_probability:.2f}입니다.')
    else:
        st.warning(f'일반 쓰레기일 확률이 높습니다.')
