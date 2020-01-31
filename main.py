"""
CNN
Convolutional Neural Net for Image/Video Classification
Developer: Zak Zadeh
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as func
import model
import dataset
import utils

""" -------------------------------------------------------------------------"""
""" Parameters """
parser = argparse.ArgumentParser(description = 'Convolutional Neural Net')
parser.add_argument('--path', type=str, default='/data/chs/dataset/', help='dataset path')
parser.add_argument('--datasetName', type=str, default='ballDrop3', help='dataset name: mnist, f_mnist, cifar10, image, ucf101, balldrop3')
parser.add_argument('--model', type=str, default='resnet3d', help='Neural net model: custom, mobilenet, resnet; default=custom')
parser.add_argument('--nEpochs', type=int, default=100, help='number of training epochs, default=1')
parser.add_argument('--startEpoch', type=int, default=0, help='number of epochs of pretrained model, default=0')
parser.add_argument('--nGPU', type=int, default=4, help='of GPUs available. Use 0 for CPU mode, default=0')
parser.add_argument('--nBatch', type=int, default=128, help='Batch size, default=16')
parser.add_argument('--nClass', type=int, default=3, help='number of classes, default=2')
parser.add_argument('--imageSize', type=int, default=64, help='image dimension (weight & Height), default=64')
parser.add_argument('--nc', type=int, default=3, help='number of image channels, default=3')
parser.add_argument('--ndf', type=int, default=64, help='Size of feature maps in Classifier, default=64')
parser.add_argument('--lr', type=float, default=0.0001, help='classifier Learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 hyperparameter for Adam optimizers, default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta1 hyperparameter for Adam optimizers, default=0.999')
params = parser.parse_args()

""" -------------------------------------------------------------------------"""
""" initialization """
trainLoader = dataset.laod(params, 'train')
testLoader  = dataset.laod(params, 'test')
device = torch.device("cuda:0" if (torch.cuda.is_available() and params.nGPU > 0) else "cpu")

""" -------------------------------------------------------------------------"""
""" Create Networks & Optimizers"""
if (params.startEpoch == 0):

    # Create the Classifier
    if (params.model == 'custom'):
        net = model.Custom(params).to(device)
    elif (params.model == 'mobilenet'):
        net = model.VGG11(params).to(device)
    elif (params.model == 'resnet'):
        net = model.ResNet(params).to(device)
    elif (params.model == 'custom3D'):
        net = model.Custom3D(params).to(device)
    elif (params.model == 'resnet3d'):
        net = model.ResNet3D(params).to(device)
    elif (params.model == 'resnet2p1d'):
        net = model.ResNet2p1D(params).to(device)

    if (device.type == 'cuda') and (params.nGPU > 1):
        net = nn.DataParallel(net, list(range(params.nGPU)))

    # Setup Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr = params.lr, betas = (params.beta1, params.beta2))

else:
    net, optimizer = utils.loadCkpt(params)

# Training Criterion
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
# criterion = nn.KLDivLoss()

""" -------------------------------------------------------------------------"""
""" Training """
# Lists to keep track of progress
trnLosses = []
testLosses = []

print("Starting Training Loop...")


for epoch in range(params.startEpoch, params.nEpochs):
    trnTotal = 0
    tstTotal = 0
    trnCorrect = 0
    tstCorrect = 0
    for i, (data, label) in enumerate(trainLoader, 0):
        net.zero_grad()

        inData = data.to(device)
        batchSize = inData.size(0)

        label = label.to(device)

        # Forward pass real batch through D
        output = net(inData)
        labelPred = torch.max(func.softmax(output, dim = 1), 1)[1]
        trnCorrect += (labelPred == label).sum().item()
        trnTotal += batchSize
        err = criterion(output, label)
        err.backward()
        optimizer.step()
        trnLosses.append(err.item())

    for i, (data, label) in enumerate(testLoader, 0):
        inData = data.to(device)
        batchSize = inData.size(0)
        label = label.to(device)

        # Forward pass real batch through D
        output = net(inData)
        labelPred = torch.max(func.softmax(output, dim = 1), 1)[1] 
        tstCorrect += (labelPred == label).sum().item()

        tstTotal += batchSize
#         errTest = criterion(output, label)
#         testLosses.append(errTest.item())

    trnAccuracy = (100 * trnCorrect) / trnTotal
    tstAccuracy = (100 * tstCorrect) / tstTotal
    print('[%d/%d][%d/%d]\tTrain Loss: %.4f\tTrain Accuracy: %.4f'
          % (epoch, params.nEpochs, i, len(trainLoader), err.item(), trnAccuracy))
    print('[%d/%d][%d/%d]\tTest Loss: %.4f\tTest Accuracy: %.4f'
          % (epoch, params.nEpochs, i, len(testLoader), err.item(), tstAccuracy))
#     utils.saveCkpt(net, optimizer, epoch, params)

""" -------------------------------------------------------------------------"""
""" Results """
## losses during training and testing.
# utils.showLoss(trnLosses, testLosses)

## Save checkpoint
# utils.saveCkpt(net, optimizer, epoch, params)
