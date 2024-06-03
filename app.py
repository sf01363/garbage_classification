
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
    2: '금속/캔',
    3: '종이',
    4: '플라스틱',
}

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 웹페이지 제목과 설명 추가
st.set_page_config(page_title="재활용품 분리 배출", page_icon="♻️", layout="wide")
st.title("♻️ 재활용품 분리 배출")
st.write("""
    이 애플리케이션은 업로드된 이미지를 분석하여 재활용 가능한 쓰레기 종류를 분류합니다.
    이미지를 업로드하고 예측 확률 임계값을 조정하여 결과를 확인하세요.
""")

# 사이드바에 설명 추가
st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #3CB371;
        }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.header("사용 방법")
st.sidebar.write("""
    1. 재활용 쓰레기 이미지를 업로드하세요 (jpg, png, jpeg).
    2. 예측 확률 임계값을 슬라이더를 통해 조정하세요.
    3. 예측 결과와 확률을 확인하세요.
""")

# 재활용 방법과 설명을 정의한 사전
recycle_methods = {
    "플라스틱": '🧴 모든 플라스틱은 깔끔하게 헹궈 줘. 🧽💦 플라스틱이 "나도 깔끔한 게 좋아!"라고 말하고 있어. 라벨도 꼭 떼 줘야 해! 😄💕 플라스틱이 "내 라벨은 필요 없어!"라고 자랑스러워해.',  # 플라스틱에 대한 설명
    "종이": '📄 기름이나 음식물 얼룩 없는 종이만 재활용할 수 있어. 🧼 종이가 "나는 기름기가 싫어!" 하고 외치는 소리가 들리지 않니? 스테이플과 클립은 미리 빼 줘! ✂️✨ 종이가 "자유를 달라!"고 외치고 있어!',  # 종이에 대한 설명
    "금속/캔": '🥫 캔은 반짝반짝 깨끗하게 헹궈서 버려 줘. 🧽💦 캔도 샤워를 좋아한단다! 라벨은 꼭 떼 줘야 해. 😊 캔은 "내 라벨은 나의 자존심!"이라고 생각하고 있어.',  # 금속/캔에 대한 설명
    "유리": '🍾 유리병도 깨끗하게 헹궈서 버려 줘. 🧴💧 유리병이 "깨끗하게 부탁해!"라고 속삭이는 것 같지 않니? 뚜껑과 마개는 빼 줘! 🥂 유리병이 "뚜껑은 나의 모자야!"라고 생각해.',  # 유리에 대한 설명
    "골판지": '📦 모든 테이프랑 스티커는 싹싹 제거해 줘! ✂️ 아니, 골판지가 패션쇼에 나가는 건 아니지만 깔끔하게 해주면 더 좋아해. 😄 그리고 젖지 않게 잘 보관해 줘. ☔️ 골판지의 천적은 물이야!'  # 골판지에 대한 설명
}

# 드롭다운 메뉴를 생성하여 선택한 재활용 방법에 대한 설명을 표시
selected_method = st.sidebar.selectbox("재활용 방법", list(recycle_methods.keys()))
if selected_method:
    st.sidebar.write(recycle_methods[selected_method])


# 여기서부터는 이미지
file = st.file_uploader('이미지를 올려주세요', type=['jpg', 'png'])

if file is None:
    st.text('이미지를 먼저 올려주세요.')
else:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True, width=0.8)
    
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
    
    # 예측된 클래스에 따라 해당 클래스의 배출 방법을 표시
    if pred_probability >= threshold:
        st.success(f'분류 결과 : 해당 재활용품이 "{pred_class}"일 확률은 {pred_probability:.2f}입니다.')
        if pred_class in recycle_methods:
            st.write(recycle_methods[pred_class])
        else:
            st.warning("해당하는 배출 방법이 없습니다.")
    else:
        st.warning(f'분류 결과 : 일반 쓰레기일 확률이 높습니다.')
