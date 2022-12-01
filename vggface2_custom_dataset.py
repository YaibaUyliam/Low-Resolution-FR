import os
import cv2
import sys
import torch
from tqdm import tqdm
from PIL import Image
import random 

from torch.utils.data import Dataset


class VGGFace2Dataset(Dataset):
    def __init__(self, root, transforms, train, logging, **kwargs):
        self._root = root
        self.transforms = transforms
        self._train = train
        self._valid_resolution = kwargs['valid_fix_resolution']
        self._algo_name = kwargs['algo_name']
        self._algo = kwargs['algo_val']
        self._classes, self._class_to_idx = self._find_classes()
        self._samples = self._make_dataset()
        self._loader = self._get_loader
        tr = 'training' if self._train else 'validation'
        logging.info(
            f'VGGFace2 custom {tr} dataset info:'
            f'\n\t\t\t\tRoot folder:       {self._root}'
            f'\n\t\t\t\tValid resolution:  {self._valid_resolution}'
        )

    def _find_classes(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self._root) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._root) if os.path.isdir(os.path.join(self._root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}       
        return classes, class_to_idx

    def _make_dataset(self):
        images = []
        dir = os.path.expanduser(self._root)
        progress_bar = tqdm(
                        sorted(self._class_to_idx.keys()),
                        desc='Making data training set' if self._train else 'Making data validation set',
                        total=len(self._class_to_idx.keys()),
                        leave=False
                    )
        for target in progress_bar:
            d = os.path.join(dir, target) #path to folder class
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self._class_to_idx[target])
                    images.append(item)
            progress_bar.update(n=1)
        progress_bar.close()
        return images

    @staticmethod
    def _get_loader(path):
        return Image.fromarray(cv2.imread(path))

    def _lower_resolution(self, img):
        w_i, h_i = img.size
        if self._train:
            size_down_list = [8, 12, 16, 20]
            size_down_choice = random.choice(size_down_list)
        else:
            size_down_choice = self._valid_resolution
        img2 = img.resize((size_down_choice, size_down_choice), self._algo)
        img2 = img2.resize((w_i, h_i), self._algo)
        return img2

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        path, label = self._samples[idx]
        img = self._loader(path)
        orig_img = self._loader(path)
        img_lr = self._lower_resolution(img)
        if self.transforms:
            img_lr = self.transforms(img_lr)
            orig_img = self.transforms(orig_img)
        return img_lr, orig_img, label
