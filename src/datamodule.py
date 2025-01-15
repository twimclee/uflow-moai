import os
from glob import glob
from pathlib import Path

import yaml
import numpy as np
import torch.utils.data
from PIL import Image

import torch
import lightning as L
from torchvision import transforms

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class UFlowDatamodule(L.LightningDataModule):
    def __init__(self, data_dir, input_size, batch_train, batch_test, image_transform, workers=1, shuffle_test=False, mode='train'):
        super().__init__()
        self.data_dir = data_dir
        self.input_size = input_size
        self.batch_train = batch_train
        self.batch_val = batch_test
        self.image_transform = image_transform
        self.shuffle_test = shuffle_test
        self.workers = workers

        if mode == 'train':
            self.train_dataset = get_dataset(self.data_dir, self.input_size, self.image_transform, mode='train')
            self.val_dataset = get_dataset(self.data_dir, self.input_size, self.image_transform, mode='valid')
        elif mode == 'valid':
            self.valtest_dataset = get_dataset(self.data_dir, self.input_size, self.image_transform, mode='valtest')
        elif mode == 'test':
            self.test_dataset = get_dataset(self.data_dir, self.input_size, self.image_transform, mode='test')

    def train_dataloader(self):
        return get_dataloader(self.train_dataset, self.batch_train, workers=self.workers)

    def val_dataloader(self):
        return get_dataloader(self.val_dataset, self.batch_val, workers=self.workers, shuffle=False)

    def valtest_dataloader(self):
        return get_dataloader(self.valtest_dataset, 1, shuffle=False)

    def test_dataloader(self):
        return get_dataloader(self.test_dataset, 1, shuffle=False)


def get_dataset(data_dir, input_size, image_transform, mode):
    return UFlowDataset(
        root=data_dir,
        input_size=input_size,
        image_transform=image_transform,
        mode=mode
    )

def get_dataloader(dataset, batch, workers=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
        worker_init_fn=worker_init_fn
    )

class UFlowDataset(torch.utils.data.Dataset):
    def __init__(self, root, input_size, image_transform, mode='train'):
        self.mean = MEAN
        self.std = STD
        self.un_normalize_transform = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.image_transform = image_transform
        self.mode = mode

        # 확장자 패턴을 추가하여 다양한 이미지 형식을 지원
        file_extensions = ["png", "jpg", "jpeg", "bmp", "tiff"]
        image_file_pattern = [os.path.join(root, "train", "good", f"*.{ext}") for ext in file_extensions]
        if mode == 'train':
            self.image_files = []
            for pattern in image_file_pattern:
                self.image_files.extend(glob(pattern))

        elif mode == 'valid':
            test_pattern = [os.path.join(root, "valid", "*", f"*.{ext}") for ext in file_extensions]
            self.image_files = []
            for pattern in test_pattern:
                self.image_files.extend(glob(pattern))
            self.image_files.sort()
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    # transforms.Normalize(self.mean.tolist(), self.std.tolist()),
                ]
            )

        elif mode == 'test':
            test_pattern = [os.path.join(root, f"*.{ext}") for ext in file_extensions]
            self.image_files = []
            for pattern in test_pattern:
                self.image_files.extend(glob(pattern))
            self.image_files.sort()

        elif mode == 'valtest':
            test_pattern = [os.path.join(root, "*", f"*.{ext}") for ext in file_extensions]
            self.image_files = []
            for pattern in test_pattern:
                self.image_files.extend(glob(pattern))
            self.image_files.sort()

    def un_normalize(self, img):
        return self.un_normalize_transform(img)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert('RGB')
        image = self.image_transform(image)

        if self.mode == 'train':
            return image

        elif self.mode == 'valid':
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("valid", "ground_truth").replace(
                        os.path.splitext(image_file)[1], "_mask.png"
                    )
                )
                target = self.target_transform(target)
            return image, target, image_file

        elif self.mode == 'test':
            target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            return image, target, image_file

        elif self.mode == 'valtest':
            target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            return image, target, image_file

    def __len__(self):
        return len(self.image_files)

def uflow_un_normalize(torch_img):
    un_normalize_transform = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
    return un_normalize_transform(torch_img)

def get_debug_images_paths(path):
    debug_images_paths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            debug_images_paths.append(os.path.join(dirpath, filename))
    return debug_images_paths