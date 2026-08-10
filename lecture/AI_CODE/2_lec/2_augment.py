from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

### 1단계 : 증강 설정하기 (주석된 옵션은 필요한 것만 선택해서 사용) ###
datagen = ImageDataGenerator(
    # rotation_range=30,          # 이미지 회전 (0~30도)
    # width_shift_range=0.2,      # 좌우 이동
    # height_shift_range=0.2,     # 상하 이동
    # shear_range=0.2,            # 전단 변형 (기울기)
    # zoom_range=0.2,             # 확대/축소
    # horizontal_flip=True,       # 좌우 반전
    # vertical_flip=True,         # 상하 반전
    brightness_range=[0.5, 1.5],# 밝기 조절
    # channel_shift_range=50.0,   # 색상 채널 이동
    # fill_mode='nearest'         # 이동/회전 후 빈 영역 처리 방법 ('constant', 'nearest', 'reflect' 등)
)


### 2단계 : 이미지 폴더 설정 ###
source_folder = 'source/train_data/right'   # 원본 이미지 폴더
save_folder = 'source/train_data/right'     # 증강 이미지 저장 폴더

os.makedirs(save_folder, exist_ok=True)

image_list = []
for f in os.listdir(source_folder):        # 폴더 안의 모든 파일 이름에 대해
    if f.endswith(('.jpg', '.png', '.jpeg')):  # 이미지 확장자면
        image_list.append(f)               # 리스트에 추가
        
print(f"총 {len(image_list)}개의 이미지에서 증강 시작")


### 3단계 : 이미지 하나씩 불러와서 5개씩 증강 ###
for img_name in image_list:
    path = os.path.join(source_folder, img_name)
    img = load_img(path)
    arr = img_to_array(img)
    arr = arr.reshape((1,) + arr.shape)

    count = 0
    for batch in datagen.flow(
        arr,
        batch_size=1,
        save_to_dir=save_folder,
        save_prefix='aug_' + os.path.splitext(img_name)[0],
        save_format='jpg'
    ):
        count += 1
        if count >= 5:
            break  # 이미지당 5개만 생성

print("증강 완료")

###################################################################
#모든 폴더를 순회하며 내부의 모든 이미지를 증강하는 방식
###################################################################
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

# 증강 설정
datagen = ImageDataGenerator(
    zoom_range=0.2,
    brightness_range=[0.5, 1.5]
)

# train 폴더 전체
base_folder = 'L_8/train_data'

# 클래스 폴더 자동 순회
for class_name in os.listdir(base_folder):

    class_path = os.path.join(base_folder, class_name)

    # 폴더만 처리
    if not os.path.isdir(class_path):
        continue

    # 이미지 목록 수집
    image_list = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    # 이미지 하나씩 증강
    for img_name in image_list:

        img_path = os.path.join(class_path, img_name)

        img = load_img(img_path)
        arr = img_to_array(img)
        arr = arr.reshape((1,) + arr.shape)

        count = 0

        for batch in datagen.flow(
            arr,
            batch_size=1,
            save_to_dir=class_path,  # 같은 클래스 폴더에 저장
            save_prefix='aug_' + os.path.splitext(img_name)[0],
            save_format='jpg'
        ):
            count += 1
            if count >= 5:
                break

print("증강 완료")
'''