import random
import cv2 as cv
import numpy
import numpy as np
import os
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from utils import cv2pil
from skimage.metrics import mean_squared_error

index_dict = {
    "lens": ["35", "105", "50_300"],
    "top_left": [1, 6],
    "top_right": [4, 5],
    "top_center": [2, 3],
    "middle": [7, 8]
}


def set_pattern(pattern, w, h, tlx, tly):
    hp, wp, _ = pattern.shape
    img_p = np.zeros((h, w, 3))
    assert w >= wp and h >= hp
    img_p[tly: tly + hp, tlx: tlx + wp, :] = pattern
    return img_p


def align_left_pattern(img, pattern):
    h, w, _ = img.shape
    hp, wp, _ = pattern.shape

    if h < hp or w < wp:
        rh = h / hp
        rw = w / wp

        if rh < rw:
            ratio = rh
        else:
            ratio = rw

        dim = (int(wp * ratio), int(hp * ratio))
        pattern = cv.resize(pattern, dim, interpolation=cv.INTER_CUBIC)
        hp, wp, _ = pattern.shape

    img_p = set_pattern(pattern, w, h, 0, 0)

    img_res = 1 - (1 - img / 255.) * (1 - img_p / 255.)
    img_res = img_res[0: hp, 0: wp, :]
    img_gt = img[0: hp, 0: wp, :]

    return np.clip(255 * img_res, 0, 255).astype(np.uint8), img_gt


def align_right_pattern(img, pattern):
    h, w, _ = img.shape
    hp, wp, _ = pattern.shape

    if h < hp or w < wp:
        rh = h / hp
        rw = w / wp

        if rh < rw:
            ratio = rh
        else:
            ratio = rw

        dim = (int(wp * ratio), int(hp * ratio))
        pattern = cv.resize(pattern, dim, interpolation=cv.INTER_CUBIC)
        hp, wp, _ = pattern.shape

    assert hp <= h and wp <= w

    img_p = np.zeros((h, w, 3), dtype=np.uint8)
    img_p[0:hp, -wp:, :] = pattern
    img_res = 1 - (1 - img / 255.) * (1 - img_p / 255.)
    img_res = img_res[0:hp, -wp:, :]
    img_gt = img[0:hp, -wp:, :]
    return np.clip(255 * img_res, 0, 255).astype(np.uint8), img_gt


def align_top_pattern(img, pattern):
    h, w, _ = img.shape
    hp, wp, _ = pattern.shape

    if h < hp or w < wp:
        rh = h / hp
        rw = w / wp

        if rh < rw:
            ratio = rh
        else:
            ratio = rw

        dim = (int(wp * ratio), int(hp * ratio))
        pattern = cv.resize(pattern, dim, interpolation=cv.INTER_CUBIC)
        hp, wp, _ = pattern.shape

    assert hp <= h and wp <= w

    lt = (w - wp) // 2
    rt = w - wp - lt
    bt = h - hp

    img_p = cv.copyMakeBorder(pattern, 0, bt, lt, rt, cv.BORDER_REFLECT)
    img_res = 1 - (1 - img / 255.) * (1 - img_p / 255.)
    img_res = img_res[0: hp, lt: lt + wp, :]
    img_gt = img[0: hp, lt: lt + wp, :]

    return np.clip(255 * img_res, 0, 255).astype(np.uint8), img_gt


def align_middle_pattern(img, pattern):
    h, w, _ = img.shape
    hp, wp, _ = pattern.shape

    if h < hp or w < wp:
        rh = h / hp
        rw = w / wp

        if rh < rw:
            ratio = rh
        else:
            ratio = rw

        dim = (int(wp * ratio), int(hp * ratio))
        pattern = cv.resize(pattern, dim, interpolation=cv.INTER_CUBIC)
        hp, wp, _ = pattern.shape

    assert hp <= h and wp <= w

    tt = (h - hp) // 2
    lt = (w - wp) // 2
    rt = w - wp - lt
    bt = h - hp - tt

    img_p = cv.copyMakeBorder(pattern, tt, bt, lt, rt, cv.BORDER_REFLECT)

    img_res = 1 - (1 - img / 255.) * (1 - img_p / 255.)
    img_res = img_res[tt: tt + hp, lt: lt + wp, :]
    img_gt = img[tt: tt + hp, lt: lt + wp, :]

    return np.clip(255 * img_res, 0, 255).astype(np.uint8), img_gt


def handle_position(alpha, img, lens_type, color_type, patterns_path):
    callbacks = [align_left_pattern, align_right_pattern, align_top_pattern, align_middle_pattern]
    idx_key = int(np.random.random() // 0.25) + 1
    position = list(index_dict.keys())[idx_key]

    case = int(np.random.random() // 0.5)
    pattern_type = index_dict[position][case]
    pattern_name = "pattern_{}_{}_{}.png".format(lens_type, pattern_type, color_type)
    pattern_path = "{}/{}".format(patterns_path, pattern_name)
    pattern = cv.imread(pattern_path)
    pattern = (alpha * pattern).astype(np.uint8)
    return callbacks[idx_key - 1](img, pattern)


class GradualImageSet(data.Dataset):
    def __init__(self, set_path, numsamples, startidx=0, aug=True, size=(512, 512), crsize=(800, 800), mode='rcrop'):
        self.num_samples = numsamples
        self.start_idx = startidx
        self.last_idx = startidx + numsamples
        self.aug = aug
        self.size = size
        self.crsize = crsize
        self.mode = mode

        im_dir = '{}/glare-free'.format(set_path)
        self.pattern_dir = '{}/flare_patterns'.format(set_path)

        self.im_list = []

        im_file_list = [f for f in os.listdir(im_dir) if f.endswith(".png")]

        for f in im_file_list:
            inp_path = os.path.join(im_dir, f)
            self.im_list.append(inp_path)

        print("Generating {} samples starting at {}".format(self.num_samples, self.start_idx))

    def __len__(self):
        return self.num_samples

    def augs(self, inp, gt, idx):
        if self.mode == 'rcrop' and idx > 0.05 * self.num_samples:
            w, h = gt.size
            if h - self.crsize[1] > 0 and w - self.crsize[0] > 0:
                ty = np.random.randint(0, h - self.crsize[0])
                tx = np.random.randint(0, w - self.crsize[1])

                gt = torchvision.transforms.functional.crop(gt, ty, tx, self.crsize[0], self.crsize[1])
                inp = torchvision.transforms.functional.crop(inp, ty, tx, self.crsize[0], self.crsize[1])

        gt = torchvision.transforms.functional.resize(gt, self.size, InterpolationMode.BICUBIC)
        inp = torchvision.transforms.functional.resize(inp, self.size, InterpolationMode.BICUBIC)

        if random.random() < 0.5:
            inp = torchvision.transforms.functional.hflip(inp)
            gt = torchvision.transforms.functional.hflip(gt)
        if random.random() < 0.5:
            inp = torchvision.transforms.functional.vflip(inp)
            gt = torchvision.transforms.functional.vflip(gt)
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            inp = torchvision.transforms.functional.rotate(inp, angle)
            gt = torchvision.transforms.functional.rotate(gt, angle)

        return inp, gt

    def __getitem__(self, index):
        idx_im = np.random.randint(0, len(self.im_list))
        inp_img = cv.imread(self.im_list[idx_im])

        # sample for lens type
        nl = np.random.random() - 0.01
        lidx = int(nl // 0.33)
        lens_name = index_dict["lens"][lidx]

        # sample for color scheme
        nc = np.random.random()
        cidx = int(nc // 0.2) + 1

        alpha = 0.5 * np.random.random() + 0.5 * (self.start_idx + index) / self.last_idx
        # sample for position
        flare_img, gt_img = handle_position(alpha, inp_img, lens_name, cidx, self.pattern_dir)
        assert flare_img.shape == gt_img.shape

        to_tensor = transforms.ToTensor()

        flare_img = cv2pil(flare_img)
        gt_img = cv2pil(gt_img)
        if not self.aug:
            return to_tensor(flare_img), to_tensor(gt_img)
        else:
            flare_sample, gt_sample = self.augs(flare_img, gt_img, index)

            return to_tensor(flare_sample), to_tensor(gt_sample)

