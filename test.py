import argparse
import math
from datetime import datetime

import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, ToTensor, transforms
from torch.utils.data import DataLoader
import model as net
from nose_pet_dataset import PetDataset

parser = argparse.ArgumentParser()
parser.add_argument('-b', "--batch_size", type=int, default=8, help='batch size for data loaders')
parser.add_argument('-l', "--load_param", type=str)
args = parser.parse_args()

device = torch.device('cuda')

model = net.XYLocationImageRegressor()
trained_params = torch.load(args.load_param)

model.load_state_dict(trained_params)

transform = Compose([
    ToTensor(),
    transforms.Resize((224, 224), antialias=True)
])

# testing data
test_set = PetDataset(img_dir='dataset/images', training=False, transform=transform)
test_loader = DataLoader(dataset=test_set, batch_size=int(args.batch_size), shuffle=False)


def test():
    start_time = datetime.now()
    model.eval()
    model.to(device)

    total_imgs = 0
    correct_pred = 0
    threshold = 0.50
    distances = []

    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for imgs, (x_labels, y_labels) in test_loader:

            # combine x and y labels
            labels = torch.stack([x_labels, y_labels], dim=1).float()

            imgs = imgs.squeeze().to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)

            total_imgs += labels.size(0)

            # calculate Euclidean distance for each point
            for i in range(0, labels.size(0)):
                x1 = labels[i].data[0].item()
                y1 = labels[i].data[1].item()
                x2 = outputs[i].data[0].item()
                y2 = outputs[i].data[1].item()

                all_labels.append((x1, y1))
                all_outputs.append((x2, y2))

                # threshold calculation
                if (x1 * (1 + threshold) > x2 > x1 * (1 - threshold)) and (y1 * (1 + threshold) > y2 > y1 * (1 - threshold)):
                    correct_pred += 1

                e_dist = math.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2))  # euclidean distance
                distances += [e_dist]


    end_time = datetime.now()
    print(f'Average Time per Image {(end_time-start_time)/total_imgs}s')

    accuracy = round((correct_pred / total_imgs) * 100, 2)
    print(f'General Accuracy: {accuracy}% for threshold: {threshold}')

    min_dist = round(min(distances), 2)
    mean_dist = round(numpy.mean(distances), 2)
    max_dist = round(max(distances), 2)
    std_dev_dist = round(numpy.std(distances), 2)

    # print accuracy statistics
    print(f'Note that all statistics are on downscaled (224x224) images')
    print(f'Min Distance: {min_dist}px')
    print(f'Mean Distance: {mean_dist}px')
    print(f'Max Distance: {max_dist}px')
    print(f'Standard Deviation of Distance: {std_dev_dist}px')


if __name__ == "__main__":
    test()
