import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from PIL import Image
from videoReader import VideoFolder, videoLoader

def loadTrain(params):
    dataRoot = params.path + params.datasetName + '/'
    name = params.datasetName
    nWorkers = 4 
    nFrames = params.nFrames
    transVid = transforms.Compose([transforms.Resize(size = (params.imageSize, params.imageSize), interpolation = Image.NEAREST), transforms.RandomRotation(90.), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), transforms.ToTensor(),])

    if (name == 'image'):
        dataset = dset.ImageFolder(root = dataroot, transform = transforms.Compose([transforms.Resize(params.imageSize),
                                       transforms.CenterCrop(params.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
        dataloader = torch.utils.data.DataLoader(dataRoot+'train/', batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
    elif (name == 'mnist'):
        transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = dset.MNIST(root = dataRoot, train = True, download = True, transform = transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
        params.nc = 1
        params.nClass = 10
    elif (name == 'f_mnist'):
        transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = dset.FashionMNIST(root = dataRoot, train = True, download = True, transform = transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
        params.nc = 1
        params.nClass = 10
    elif (name == 'cifar10'):
        transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = dset.CIFAR10(root = dataRoot, train = True, download = True, transform = transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
        params.nClass = 10
    elif (name == 'kth'):
        dataset = VideoFolder(video_root = dataRoot+'train/', video_ext = '.png', nframes = nFrames, loader = videoLoader, transform = transVid)
        params.nc = 1
        params.nClass = 6
        params.isInVideo = True
    elif (name == 'ucf101'):
        dataset = VideoFolder(video_root = dataRoot+'train/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transVid)
#         dataset = dset.UCF101(dataRoot, annotation_path = params.path+'/list/', frames_per_clip=1, train=True, transform=None, num_workers = nWorkers)
        params.nClass = 101
        params.isInVideo = True
    elif (name == 'hmdb51'):
        dataset = VideoFolder(video_root = dataRoot+'train/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transVid)
        params.nClass = 51
        params.isInVideo = True
    elif (name == 'ballDrop3'):
        dataRoot = '/data/chs/dataset/' + params.datasetName + '/'
        dataset = VideoFolder(video_root = dataRoot+'train/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transVid)
        params.nClass = 3
        params.isInVideo = True
    else:
        print('Error : Wrong Dataset !')
    dataloader = data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
    return dataloader

def loadTest(params):
    dataRoot = params.path + params.datasetName + '/'
    name = params.datasetName
    nWorkers = 3
    nFrames = params.nFrames
    transVid = transforms.Compose([transforms.Resize(size = (params.imageSize, params.imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])

    if (name == 'image'):
        dataset = dset.ImageFolder(root = dataroot, transform = transforms.Compose([transforms.Resize(params.imageSize),
                                       transforms.CenterCrop(params.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
        dataloader = torch.utils.data.DataLoader(dataRoot+'train/', batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
    elif (name == 'mnist'):
        transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = dset.MNIST(root = dataRoot, train = False, download = True, transform = transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
        params.nc = 1
        params.nClass = 10
    elif (name == 'f_mnist'):
        transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = dset.FashionMNIST(root = dataRoot, train = False, download = True, transform = transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
        params.nc = 1
        params.nClass = 10
    elif (name == 'cifar10'):
        transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = dset.CIFAR10(root = dataRoot, train = False, download = True, transform = transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
        params.nClass = 10
    elif (name == 'kth'):
        dataset = VideoFolder(video_root = dataRoot+'test/', video_ext = '.png', nframes = nFrames, loader = videoLoader, transform = transVid)
        params.nc = 1
        params.nClass = 6
        params.isInVideo = True
    elif (name == 'ucf101'):
        dataset = VideoFolder(video_root = dataRoot+'test/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transVid)
        params.nClass = 101
        params.isInVideo = True
    elif (name == 'hmdb51'):
        dataset = VideoFolder(video_root = dataRoot+'test/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transVid)
        params.nClass = 51
        params.isInVideo = True
    elif (name == 'ballDrop3'):
        dataRoot = '/data/chs/dataset/' + params.datasetName + '/'
        dataset = VideoFolder(video_root = dataRoot+'test/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transVid)
        params.nClass = 3
        params.isInVideo = True
    else:
        print('Error : Wrong Dataset !')
    dataloader = data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
    return dataloader
