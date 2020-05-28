import random
import os
import re
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

from torchvision.transforms import functional as F


class CasiaSurfDataset(Dataset):
    def __init__(self, protocol: int, dir: str = 'data/CASIA_SURF', mode: str = 'train', depth=True, ir=True,
                 transform=None):
        self.dir = dir
        self.mode = mode
        submode = {'train': 'train', 'dev': 'dev_ref',
                   'test': 'test_res'}[mode]
        file_name = f'4@{protocol}_{submode}.txt'
        with open(os.path.join(dir, file_name), 'r') as file:
            lines = file.readlines()
            self.items = []
            for line in lines:
                if self.mode == 'train':
                    img_name, label = tuple(line[:-1].split(' '))
                    self.items.append(
                        (self.get_all_modalities(img_name, depth, ir), label))

                elif self.mode == 'dev':
                    folder_name, label = tuple(line[:-1].split(' '))
                    profile_dir = os.path.join(
                        self.dir, folder_name, 'profile')
                    for file in os.listdir(profile_dir):
                        img_name = os.path.join(folder_name, 'profile', file)
                        self.items.append(
                            (self.get_all_modalities(img_name, depth, ir), label))

                elif self.mode == 'test':
                    folder_name = line[:-1].split(' ')[0]
                    profile_dir = os.path.join(
                        self.dir, folder_name, 'profile')
                    for file in os.listdir(profile_dir):
                        img_name = os.path.join(folder_name, 'profile', file)
                        self.items.append(
                            (self.get_all_modalities(img_name, depth, ir), -1))

        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_names, label = self.items[idx]
        images = []
        for img_name in img_names:
            img_path = os.path.join(self.dir, img_name)
            img = Image.open(img_path).convert(
                'RGB' if 'profile' in img_path else 'L')
            if self.transform:
                random.seed(idx)
                img = self.transform(img)
            images += [img]

        return torch.cat(images, dim=0), int(label)

    def get_video_id(self, idx: int):
        img_name = self.items[idx][0]
        return re.search(rf'(?P<id>{self.mode}/\d+)', img_name).group('id')

    def get_all_modalities(self, img_path: str, depth: bool = True, ir: bool = True) -> list:
        result = [img_path]
        if depth:
            result += [re.sub('profile', 'depth', img_path)]
        if ir:
            result += [re.sub('profile', 'ir', img_path)]

        return result


class NonZeroCrop(object):
    """Cut out black regions.
    """

    def __call__(self, img):
        arr = np.asarray(img)
        pixels = np.transpose(arr.nonzero())
        if len(arr.shape) > 2:
            pixels = pixels[:, :-1]
        top = pixels.min(axis=0)
        h, w = pixels.max(axis=0) - top
        return F.crop(img, top[0], top[1], h, w)
