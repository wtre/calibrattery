# -*- coding: utf-8 -*-
"""
Training a Classifier
=====================
This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.
Now you might be thinking,
What about data?
----------------
Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.
-  For images, packages such as Pillow, OpenCV are useful
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful
Specifically for vision, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.
This provides a huge convenience and avoids writing boilerplate code.
For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.
.. figure:: /_static/img/cifar10.png
   :alt: cifar10
   cifar10
Training an image classifier
----------------------------
We will do the following steps in order:
1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

########################################################################
# This part of code is for testing baseline networks without any pruning.

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# libraries added
import argparse
import torchvision.datasets as datasets
import os
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping_deprecated


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main_two(args, ITE=0):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    #                                           shuffle=True, num_workers=2)
    #
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                          shuffle=False, num_workers=2)

    trainvalset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    trainset_size = int((1-args.valid_size) * len(trainvalset))
    valset_size = len(trainvalset) - trainset_size

    trainset, valset = torch.utils.data.random_split(trainvalset, [trainset_size, valset_size])
    testset = datasets.CIFAR10('../data', train=False, transform=transform)\

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)


    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    ########################################################################
    # Let us show some of the training images, for fun.
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


    ########################################################################
    # 2. Define a Convolutional Neural Network
    # net = Net()
    from archs.cifar10 import minivgg
    model = minivgg.conv4().to(device)

    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.


    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    for big_iter in range(1):
        early_stopping = EarlyStopping_deprecated(patience=99, verbose=True)
        # model = minivgg.conv4().to(device)        ############# Reinit here if you need to reset ##############

        for epoch in range(33):  # loop over the dataset multiple times

            running_loss = 0.0
            model.train()
            for i, (inputs, labels) in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:    # print every 2000 mini-batches
                    print('    [%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

                train_losses.append(loss.item())

            # Validate the model!!!
            model.eval()
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            print_msg = (f'[{epoch:2d}/40] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # test loop!
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'    Accuracy of the network on the 10000 test images: {100 * correct / total:2.2f} %')

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    print('Finished Training')

    ########################################################################
    # Let's quickly save our trained model:
    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)

    ########################################################################
    # See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_
    # for more details on saving PyTorch models.
    #
    # 5. Test the network on the test data
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # We have trained the network for 2 passes over the training dataset.
    # But we need to check if the network has learnt anything at all.
    #
    # We will check this by predicting the class label that the neural network
    # outputs, and checking it against the ground-truth. If the prediction is
    # correct, we add the sample to the list of correct predictions.
    #
    # Okay, first step. Let us display an image from the test set to get familiar.

    # dataiter = iter(testloader)
    # images, labels = dataiter.next()
    #
    # # print images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    print(' Done!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.0003, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--valid_size", default=0.08334, type=float, help="Size of validation set")

    args = parser.parse_args()
    main_two(args)