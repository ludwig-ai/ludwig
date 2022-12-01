# coding=utf-8

import os

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
import argparse

from models.central import avmnist
from datasets import avmnist as avmnist_dataset

device_ids = [0, 1]
main_device_idx = 0
batch_size = 64
n_epochs = 50

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--channels', type=int, default=3, help='The channels of the input data.')
parser.add_argument('--datadir', type=str, default='/workspace/mfas/datasets/avmnist',
                    help='The data directory of avmnist.')
parser.add_argument('--workers_num', type=int, default=2, help='The num of workers for data loader.')
parser.add_argument('--num_outputs', type=int, default=10, help='The dimension of the model\'s output.')
parser.add_argument('--learning_rate', type=float, default=1e-1, help='The learning rate of the optimizer.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='The weight decay of optimizer.')
args = parser.parse_args()

transform = transforms.Compose([avmnist_dataset.ToTensor()])

train_dataset = avmnist_dataset.AVMnist(root_dir=args.datadir, transform=transform, stage='train', modal_separate=True,
                                        modal='image')
val_dataset = avmnist_dataset.AVMnist(root_dir=args.datadir, transform=transform, stage='train', modal_separate=True,
                                      modal='image')
test_dataset = avmnist_dataset.AVMnist(root_dir=args.datadir, transform=transform, stage='test', modal_separate=True,
                                       modal='image')

train_idxs = list(range(0, 55000))
valid_idxs = list(range(55000, 60000))

train_subset = Subset(train_dataset, train_idxs)
valid_subset = Subset(val_dataset, valid_idxs)

train_loader = DataLoader(dataset=train_subset, batch_size=batch_size * len(device_ids), shuffle=False,
                          num_workers=args.workers_num)
val_loader = DataLoader(dataset=valid_subset, batch_size=batch_size * len(device_ids), shuffle=False,
                        num_workers=args.workers_num)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size * len(device_ids), shuffle=True,
                         num_workers=args.workers_num)

model = avmnist.GP_LeNet(args=args, in_channels=1)

model = torch.nn.DataParallel(model, device_ids=device_ids)

# put the model on the main device
model = model.cuda(device=device_ids[main_device_idx])

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    validate_correct = 0
    testing_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    print('train')
    for data in tqdm(train_loader):
        X_train, y_train = data['image'], data['label']
        # 注意数据也是放在主设备
        X_train, y_train = X_train.cuda(device=device_ids[main_device_idx]), y_train.cuda(
            device=device_ids[main_device_idx])

        outputs = model(X_train)
        _, pred = torch.max(outputs[0].detach(), 1)
        optimizer.zero_grad()
        loss = cost(outputs[0], y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
    print('validate')
    for data in tqdm(val_loader):
        X_val, y_val = data['image'], data['label']
        X_val, y_val = X_val.cuda(device=device_ids[main_device_idx]), y_val.cuda(device=device_ids[main_device_idx])
        outputs = model(X_val)
        _, pred = torch.max(outputs[0].detach(), 1)
        validate_correct += torch.sum(pred == y_val.detach())
    print('test')
    for data in tqdm(test_loader):
        X_test, y_test = data['image'], data['label']
        X_test, y_test = X_test.cuda(device=device_ids[main_device_idx]), y_test.cuda(
            device=device_ids[main_device_idx])
        outputs = model(X_test)
        _, pred = torch.max(outputs[0].data, 1)
        testing_correct += torch.sum(pred == y_test.detach())
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Validate Accuracy is :{:.4f}%, Test Accuracy is:{:.4f}%".format(
        torch.true_divide(running_loss, len(train_subset)),
        torch.true_divide(100 * running_correct, len(train_subset)),
        torch.true_divide(100 * validate_correct, len(valid_subset)),
        torch.true_divide(100 * testing_correct, len(test_dataset)))
    )

torch.save(model.state_dict(), './Checkpoints/AVMNIST/image_model.pkl')
