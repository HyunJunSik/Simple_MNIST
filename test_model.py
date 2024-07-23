from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


from PIL import Image
import numpy as np

# 이미지 만들기
# 픽셀을 28 by 28로 만들어서, 그 안에 숫자 아무거나 그리기 (0~9)
img = Image.open("data.png").convert("L")

# img = np.resize(img, (1, 784))

# test_data = ((np.array(img) / 255) - 1) * -1

test_data = np.resize(np.array(img), (2, 28, 28, 1))
test_data = test_data/255.0

model = load_model("model.h5")

res = model.predict(test_data)

print(f"prediction : {res}")
