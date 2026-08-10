from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# 경로
train_dir = 'source/train_data'
val_dir   = 'source/val_data'
test_dir  = 'source/test_data'

# 1) 제너레이터 (증강 X, 정규화만)
train_gen = ImageDataGenerator(rescale=1./255)
val_gen   = ImageDataGenerator(rescale=1./255)
test_gen  = ImageDataGenerator(rescale=1./255)

# 2) 디렉토리에서 로드
# - class_mode='categorical' → 원-핫 자동
# - shuffle: train만 True, val/test는 False 추천(재현성/평가 정확도)
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 3) CNN 모델
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

# 4) 컴파일
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5) 학습: validation은 val_data
model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# 6) 최종 테스트 평가 (학습에 사용하지 않은 test로만)
test_loss, test_acc = model.evaluate(test_data, verbose=0)
print(f"TEST 결과: loss={test_loss:.4f}, acc={test_acc:.4f}")

# 7) 저장
model.save('2_lec/cnn_model_gen_noaug.h5')
print("모델 저장 완료: 2_lec/cnn_model_gen_noaug.h5")