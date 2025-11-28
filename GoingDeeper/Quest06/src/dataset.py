import re
import six
import math
import lmdb
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class MJDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        label_converter,
        img_size=(100, 32),
        max_text_len=22,
        character="",
        ratio=1.0, # 데이터 사용 비율 파라미터 추가
        augmentations=None # Augmentation 파라미터 추가
    ):
        super().__init__()
        self.dataset_path = dataset_path # 경로만 저장
        self.label_converter = label_converter
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.character = character
        self.env = None # 처음엔 None으로 설정
        self.ratio = ratio
        self.augmentations = augmentations

        # 데이터 개수만 미리 파악하기 위해 잠시 열었다가 닫음
        env = lmdb.open(dataset_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            num_samples_total = int(txn.get("num-samples".encode()))
            self.num_samples = int(num_samples_total * self.ratio)
            self.index_list = [idx + 1 for idx in range(self.num_samples)]
        env.close()

        print(f"Total {num_samples_total} -> Used {self.num_samples} ({self.ratio*100:.1f}%)")

    def _init_db(self):
        # 실제 데이터를 읽을 때 연결 (각 워커별로 별도 연결 생성됨)
        self.env = lmdb.open(self.dataset_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        index = self.index_list[idx]
        with self.env.begin(write=False) as txn:
            label_key = f"label-{index:09d}".encode()
            label = txn.get(label_key).decode("utf-8")

            img_key = f"image-{index:09d}".encode()
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img_pil = Image.open(buf).convert("RGB")
            except IOError:
                img_pil = Image.new("RGB", self.img_size)
                label = "-"

        orig_w, orig_h = img_pil.size
        target_width = min(int(orig_w * self.img_size[1] / orig_h), self.img_size[0])
        target_img_size = (target_width, self.img_size[1])
        img_pil = img_pil.resize(target_img_size)

        img = np.array(img_pil)

        if self.augmentations:
            transformed = self.augmentations(image=img)
            img = transformed['image']

        img = img.transpose(2, 0, 1)

        # 이미지 정규화 (0~1)
        img = img.astype(np.float32) / 255.0

        padded_img = np.zeros((3, self.img_size[1], self.img_size[0]), dtype=np.float32)
        c, h, w = img.shape
        padded_img[:, :h, :w] = img

        label = label.upper()
        out_of_char = f"[^{self.character}]"
        label = re.sub(out_of_char, "", label)
        label = label[: self.max_text_len]

        encoded_label = self.label_converter.encode(label)

        return padded_img, encoded_label, len(encoded_label), label

def collate_fn(batch):
    imgs, encoded_labels, label_lens, raw_labels = zip(*batch)

    imgs_tensor = torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32)

    max_len = max(label_lens)
    labels_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, label_arr in enumerate(encoded_labels):
        length = label_lens[i]
        labels_padded[i, :length] = torch.tensor(label_arr, dtype=torch.long)

    batch_size = imgs_tensor.size(0)
    # 모델 출력 시퀀스 길이에 맞춰 46으로 수정 (모델 구조 변경 반영)
    input_length = torch.full(size=(batch_size,), fill_value=46, dtype=torch.long)
    label_length = torch.tensor(label_lens, dtype=torch.long)

    return (
        imgs_tensor,
        labels_padded,
        input_length,
        label_length,
        raw_labels,
    )
