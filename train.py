import argparse
import datetime
import os

import tensorflow as tf
import numpy as np
from keras import applications, layers
import model as net
import torch.optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader as DataLoader
import FileIO
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, transforms
import rescaler
from nose_pet_dataset import PetDataset


class EuclideanDistanceLoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(torch.sum((pred - target) ** 2, dim=1)).mean()


# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', "--epochs", type=int, default=10, help='number of epochs for training')
parser.add_argument('-b', "--batch_size", type=int, default=8, help='batch size for data loaders')
parser.add_argument('-p', "--loss_plot", type=str, help='loss plot file')
parser.add_argument('-s', "--save_model", type=str, help='location to save model')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device("cpu")

unscaled_train_dict = FileIO.file_to_dict('dataset/train_noses.3.txt')
scaled_train_dict = rescaler.original_to_scaled(unscaled_train_dict)
FileIO.dict_to_file(scaled_train_dict, 'downscaled_train_noses.3.txt')

unscaled_test_dict = FileIO.file_to_dict('dataset/test_noses.txt')
scaled_test_dict = rescaler.original_to_scaled(unscaled_test_dict)
FileIO.dict_to_file(scaled_test_dict, 'downscaled_test_noses.txt')

transform = Compose([
    ToTensor(),
    transforms.Resize((224, 224), antialias=True)
])

train_dataset = PetDataset(img_dir='dataset/images', training=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=int(args.batch_size), shuffle=True)

# validation
val_dataset = PetDataset(img_dir='dataset/images', training=False, transform=transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=int(args.batch_size), shuffle=False)

model = net.XYLocationImageRegressor()

learning_rate = 0.01
weight_decay = 1e-5
model_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

step_size = 15
gamma = 0.8
model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=step_size, gamma=gamma)

loss_function = EuclideanDistanceLoss()


def train():
    model.to(device)
    epoch_losses_train = []
    epoch_losses_val = []
    validate = True

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch #{epoch}, Start Time: {datetime.datetime.now()}')
        loss_train = 0

        # training
        model.train()
        for imgs, (x_labels, y_labels) in train_loader:
            # combine x and y labels
            labels = torch.stack([x_labels, y_labels], dim=1).float()

            # send imgs to model for processing
            imgs = imgs.squeeze().to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)

            # calculate losses
            loss = loss_function(outputs, labels)
            loss_train += loss.item()

            # backpropagation
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

        epoch_losses_train += [loss_train / len(train_loader)]
        print(f'Training Epoch {epoch} Loss: {epoch_losses_train[epoch - 1]}')
        model_scheduler.step()

        # validating
        if validate:
            loss_val = 0
            model.eval()
            with torch.no_grad():
                for imgs, (x_labels, y_labels) in val_loader:
                    # combine x and y labels
                    labels = torch.stack([x_labels, y_labels], dim=1).float()

                    # send imgs to model for processing
                    imgs = imgs.squeeze().to(device=device)
                    labels = labels.to(device=device)
                    outputs = model(imgs)

                    # calculate losses
                    loss = loss_function(outputs, labels)
                    loss_val += loss.item()

            epoch_losses_val += [loss_val / len(val_loader)]
            print(f'Validation Epoch {epoch} Loss: {epoch_losses_val[epoch - 1]}')

        print('\n')

    torch.save(model.state_dict(), args.save_model)
    generate_loss_plot_with_val(epoch_losses_train, epoch_losses_val, args.loss_plot, show_plot=True)


def generate_loss_plot_with_val(train_loss, val_loss, file_loc, show_plot=False):  # loss plot with validation
    epochs = list(range(1, len(train_loss) + 1))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.savefig(file_loc)
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    train()
