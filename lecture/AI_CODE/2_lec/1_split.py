import os
import shutil
import random

# 원본 폴더
source_dir = 'source/data'

train_dir = 'source/train_data'
val_dir = 'source/val_data'
test_dir = 'source/test_data'

# 분할 비율
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)  # 재현성 고정

# 클래스 폴더 순회
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = []
    for f in os.listdir(class_path):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            images.append(f)

    random.shuffle(images)

    total = len(images) 
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # 디렉토리 생성
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # 파일 복사
    for img in train_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(train_dir, class_name, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(val_dir, class_name, img))

    for img in test_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(test_dir, class_name, img))

print("train / val / test 분리 완료")