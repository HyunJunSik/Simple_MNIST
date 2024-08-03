from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np

# 이미지 로드 및 전처리
img = Image.open("data.png").convert("L")

# 이미지 크기 조정 (28x28) 및 정규화 (0~1 범위로)
img = img.resize((28, 28))
img_array = np.array(img) / 255.0

# 모델 입력 형태에 맞추기 위해 차원 추가
test_data = np.expand_dims(img_array, axis=(0, -1))

# 모델 로드
model = load_model("model.h5")

# 예측 수행
res = model.predict(test_data)

# 예측 결과 출력
predicted_class = np.argmax(res)
print(f"짜잔 제가 예측한 숫자는 {predicted_class} 입니다!")