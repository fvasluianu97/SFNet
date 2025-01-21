import os
import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from random import random, choice
from PIL import Image
from torchvision.transforms import InterpolationMode


class ImageSet(data.Dataset):
    def __init__(self, set_path, set_type, aug=True, size=(512, 512), mode='rcrop'):
        in_path_dir = '{}/{}/lens-flare'.format(set_path, set_type)
        gt_path_dir = '{}/{}/flare-free'.format(set_path, set_type)

        self.mode = mode
        self.size = size
        self.aug = aug

        self.gt_list = []
        self.inp_list = []
        self.num_samples = 0

        file_list = [f for f in os.listdir(in_path_dir) if f.endswith(".png")]

        for f in file_list:
            inp_path = os.path.join(in_path_dir, f)
            gt_path = os.path.join(gt_path_dir, f)

            self.inp_list.append(inp_path)
            self.gt_list.append(gt_path)
            self.num_samples += 1

        assert len(self.inp_list) == len(self.gt_list)
        assert len(self.inp_list) == self.num_samples

    def __len__(self):
        return self.num_samples

    def augs(self, inp, gt):
        if self.mode == 'rcrop':
            w, h = gt.size
            tl = np.random.randint(0, h - self.size[0])
            tt = np.random.randint(0, w - self.size[1])

            gt = torchvision.transforms.functional.crop(gt, tt, tl, self.size[0], self.size[1])
            inp = torchvision.transforms.functional.crop(inp, tt, tl, self.size[0], self.size[1])
        else:
            gt = torchvision.transforms.functional.resize(gt, self.size, InterpolationMode.BICUBIC)
            inp = torchvision.transforms.functional.resize(inp, self.size, InterpolationMode.BICUBIC)

        if random() < 0.5:
            inp = torchvision.transforms.functional.hflip(inp)
            gt = torchvision.transforms.functional.hflip(gt)
        if random() < 0.5:
            inp = torchvision.transforms.functional.vflip(inp)
            gt = torchvision.transforms.functional.vflip(gt)
        if random() < 0.5:
            angle = choice([90, 180, 270])
            inp = torchvision.transforms.functional.rotate(inp, angle)
            gt = torchvision.transforms.functional.rotate(gt, angle)

        return inp, gt

    def __getitem__(self, index):
        inp_data = Image.open(self.inp_list[index])
        gt_data = Image.open(self.gt_list[index])

        to_tensor = transforms.ToTensor()

        if self.aug:
            inp_data, gt_data = self.augs(inp_data, gt_data)
        else:
            inp_data = torchvision.transforms.functional.resize(inp_data, self.size, InterpolationMode.BICUBIC)
            gt_data = torchvision.transforms.functional.resize(gt_data, self.size, InterpolationMode.BICUBIC)

        return to_tensor(inp_data), to_tensor(gt_data)
