import os
import cv2
from torch.utils.data import Dataset
import rescaler


class PetDataset(Dataset):
    def __init__(self, img_dir, training=True, transform=None):

        self.img_dir = img_dir
        self.training = training

        if self.training:
            self.mode = 'train'
            self.label_file = 'dataset/train_noses.3.txt'
        else:
            self.mode = 'test'
            self.label_file = 'dataset/test_noses.txt'

        self.transform = transform
        self.num = 0
        self.img_files = []

        f = open(self.label_file)
        for line in f.readlines():
            img_file = line.split(',')[0]
            self.img_files += [img_file]

        self.max = len(self)

    def prepare(self, image, label):
        return self.transform()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = self.transform(cv2.imread(img_path, cv2.IMREAD_COLOR))

        label = []
        f = open(self.label_file)
        for line in f.readlines():
            unscaled_dict = {}
            if self.img_files[idx] in line:
                split_string = line.split(',')
                x = int(split_string[1][2:])
                y = int(split_string[2][1:-3])

                # apply label scaling (for downscaling to 128x128 image)
                unscaled_dict[self.img_files[idx]] = (x, y)
                scaled_dict = rescaler.original_to_scaled(unscaled_dict)

                label = list(scaled_dict[self.img_files[idx]])

        return image, label

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if self.num >= self.max:
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num - 1)
