import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt

# 1. MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 전처리
# CNN 입력 형식에 맞게 차원 추가 (28x28x1) 및 정규화
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# one-hot 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),  # 추가

    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),  # 추가

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),   # 추가
    Dense(10, activation='softmax')
])

# 4. 모델 경로 지정
model_path = "L_7/mnist_cnn_model.h5"

# 5. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. 모델 학습
history = model.fit(x_train, y_train, epochs=20, batch_size=128,
                    validation_split=0.1, verbose=1)
# 모델 저장
model.save(model_path)

# 7. 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ 테스트 정확도: {test_acc:.4f}")

# 8. 예측 결과 확인
predictions = model.predict(x_test[:5])
for i, pred in enumerate(predictions):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"predict: {tf.argmax(pred).numpy()}, answer: {tf.argmax(y_test[i]).numpy()}")
    plt.axis('off')
    plt.show()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.show()

