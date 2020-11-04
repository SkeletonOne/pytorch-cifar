'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from tqdm import tqdm
from models import *
from utils import progress_bar
import numpy as np
import time
import copy

# Data
def load_data(dataset):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

# Training
def train(net, optimizer, criterion, dataloader, epoch):
    # print('\nTraining Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    datacount = len(dataloader)
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        # move the data onto the device
        inputs, targets = inputs.to(device), targets.to(device)
        # clear the previous grad 
        optimizer.zero_grad()
        # compute model outputs and loss
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # after computing gradients based on current batch loss,
        # apply them to parameters
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # write to tensorboard
        writer.add_scalar('train/loss', train_loss/(batch_idx+1), (datacount * (epoch+1)) + (batch_idx+1))
        writer.add_scalar('train/accuracy', 100.*correct/total, (datacount * (epoch+1)) + (batch_idx+1))

        progress_bar(batch_idx, len(dataloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Testing
def test(net, criterion, dataloader, epoch):
    # Set the model into test mode
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    datacount = len(dataloader)
    
    # check global variable `best_accuracy`
    global best_accuracy

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # log the test_loss
            writer.add_scalar('test/loss', test_loss/(batch_idx+1), (datacount * (epoch+1)) + (batch_idx+1))
            writer.add_scalar('test/accuracy', 100.*correct/total, (datacount * (epoch+1)) + (batch_idx+1))

            progress_bar(batch_idx, len(dataloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    current_loss = test_loss/len(dataloader)
    # save checkpoint
    acc = 100. * correct/total
    if acc > best_accuracy:
        print("Saving the model.....")
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer' : optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        net_name = net.__class__.__name__
        save_path = "./checkpoints/{}_acc_{:.3f}_loss_{:.3f}.pth".format(net_name, acc, current_loss)
        torch.save(state, save_path)
        
        best_accuracy = acc

def train_and_evaluate(net, trainloader, testloader, optimizer, scheduler, criterion, total_epochs, start_epoch):
    # Without +1: 0~299; with +1: 1~300
    for epoch in range(start_epoch + 1, total_epochs + 1):

        # Run one epoch for both train and test
        print("Epoch {}/{}".format(epoch, total_epochs), "| Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # compute number of batches in one epoch(one full pass over the training set)
        train(net, optimizer, criterion, trainloader, epoch)
        lr_ = scheduler.get_last_lr()
        print("Learning rate: ", torch.tensor(lr_))        
        writer.add_scalar('Learning rate', torch.tensor(lr_))
        
        scheduler.step()

        # Evaluate for one epoch on test set
        test(net, criterion, testloader, epoch)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--net', default='resnet18', type=str, help='network used for training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
    parser.add_argument('--checkpoint', default=None, help='The checkpoint file (.pth)')
    parser.add_argument('--epochs', default=300, help='The number of training epochs')

    args = parser.parse_args()


    trainloader, testloader = load_data('cifar10')
    # setup device for training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # setup Tensorboard file path
    writer = SummaryWriter('./summarys/resnet50')
    # Configure the Network

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    net = ResNet50()
    net = net.to(device)

    # Define loss, optimizer, lr scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
    # The milestones mean update the lr AFTER the milestone epoch
    scheduler = MultiStepLR(optimizer, milestones=[2, 3, 4], gamma=0.1)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        #
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        #
    else:
        print('==> Starting from scratch..')
        start_epoch = 0
    

    # Setup best accuracy for comparing and model checkpoints
    best_accuracy = 0.0


    # print summary of model
    # summary(net, (3, 32, 32))
    # Setup the loss function

    train_and_evaluate(net, trainloader, testloader, optimizer, scheduler, criterion, total_epochs=args.epochs, start_epoch=start_epoch)

    writer.close()
