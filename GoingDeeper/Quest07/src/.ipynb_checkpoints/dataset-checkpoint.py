import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.io import imread
from PIL import Image

class KittiDataset(Dataset):
    def __init__(self, dir_path, multiplier=1, augmentation=None, is_train=True):
        self.dir_path = dir_path
        self.multiplier = multiplier
        self.augmentation = augmentation
        self.is_train = is_train
        self.output_size = (224, 224) # Default size, can be adjusted
        
        self.data = self.load_dataset()
        
        if self.is_train:
            self.data = self.data * self.multiplier

    def load_dataset(self):
        # kitti dataset에서 필요한 정보(이미지 경로 및 라벨)를 directory에서 확인하고 로드하는 함수입니다.
        input_images = sorted(glob.glob(os.path.join(self.dir_path, "image_2", "*.png")))
        label_images = sorted(glob.glob(os.path.join(self.dir_path, "semantic", "*.png")))

        # Label 이미지가 없는 경우 (Test Dataset인 경우) 처리
        if len(label_images) == 0:
            self.has_labels = False
            return input_images
        else:
            self.has_labels = True
            assert len(input_images) == len(label_images)
            return list(zip(input_images, label_images))

    def __len__(self):
        # Dataset의 length로서 전체 dataset 크기를 반환합니다.
        return len(self.data)

    def encode_mask(self, mask):
        '''
        30여개의 KITTI 클래스를 5개(0~4)의 그룹으로 변환합니다.
        0: Background (Others)
        1: Vehicle (Car, Truck, etc.)
        2: Nature (Vegetation, Terrain)
        3: Safety (Traffic light, Sign, Guard rail)
        4: Road
        '''
        mask = mask.astype(np.int64)
        new_mask = np.zeros_like(mask, dtype=np.int64)

        # 1: Vehicle (Ego vehicle, Car, Truck, Bus, Caravan, Trailer, Train, Motorcycle, Bicycle)
        vehicle_ids = [1, 26, 27, 28, 29, 30, 31, 32, 33]
        for vid in vehicle_ids:
            new_mask[mask == vid] = 1

        # 2: Nature (Vegetation, Terrain)
        nature_ids = [21, 22]
        for nid in nature_ids:
            new_mask[mask == nid] = 2

        # 3: Safety (Guard rail, Traffic Light, Traffic Sign)
        safety_ids = [14, 19, 20]
        for sid in safety_ids:
            new_mask[mask == sid] = 3

        # 4: Road
        new_mask[mask == 7] = 4

        return new_mask

    def __getitem__(self, index):
        # 입력과 출력을 만듭니다.
        # 입력은 resize 및 augmentation이 적용된 input image이고
        # 출력은 semantic label입니다.
        if self.has_labels:
            input_img_path, output_path = self.data[index]
            _output = imread(output_path)
            _output = self.encode_mask(_output)
        else:
            input_img_path = self.data[index]
            # Label이 없으면 Dummy Mask 생성 (0으로 채움)
            _output = np.zeros(self.output_size, dtype=np.int64)

        _input = imread(input_img_path)

        data = {
            "image": _input,
            "mask": _output,
        }

        if self.augmentation:
            augmented = self.augmentation(**data)
            _input = augmented["image"] / 255.0  # Normalize
            _output = augmented["mask"]

        return (
            torch.tensor(_input, dtype=torch.float32).permute(2, 0, 1),  # (C, H, W)
            torch.tensor(_output, dtype=torch.long)  # (H, W) - CrossEntropyLoss 용
        )

    def shuffle_data(self):
        # 한 epoch가 끝나면 실행되는 함수입니다. 학습 중인 경우에 데이터를 random shuffle합니다.
        if self.is_train:
            np.random.shuffle(self.data)


