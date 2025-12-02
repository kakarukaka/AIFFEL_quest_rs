from albumentations import  HorizontalFlip, RandomSizedCrop, RandomBrightnessContrast, Compose, OneOf, Resize

def build_augmentation(is_train=True, width=224, height=224):
  if is_train:    # 훈련용 데이터일 경우
    return Compose([
                    HorizontalFlip(p=0.5),    # 50%의 확률로 좌우대칭
                    RandomSizedCrop(         # 50%의 확률로 RandomSizedCrop
                        min_max_height=(300, 370),
                        w2h_ratio=370/1242,
                        height=height,
                        width=width,
                        size=(height, width),
                        p=0.5
                        ),
                    RandomBrightnessContrast(p=0.5), # 50%의 확률로 밝기 및 대비 조절
                    Resize(              # 입력이미지를 지정된 크기로 resize
                        width=width,
                        height=height
                        )
                    ])
  return Compose([      # 테스트용 데이터일 경우에는 지정된 크기로 resize만 수행합니다.
                Resize(
                    width=width,
                    height=height
                    )
                ])