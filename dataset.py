import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from videoReader import VideoFolder, videoLoader

def laod(params, mode):
    name = params.datasetName
    dataroot = params.path + name + '/'
    nWorkers = 3
    imageSize = params.imageSize
    nFrames = 4
    
    if (mode == 'train'):
        if (name == 'mnist'):
            transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset = dset.MNIST(root = dataroot, train = True, download = True, transform = transform)
            params.nc = 1
            params.nClass = 10
        elif (name == 'f_mnist'):
            transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            dataset = dset.FashionMNIST(root = dataroot, train = True, download = True, transform = transform)
            params.nc = 1
            params.nClass = 10
        elif (name == 'cifar10'):
            transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset = dset.CIFAR10(root = dataroot, train = True, download = True, transform = transform)
            params.nClass = 10
        elif (name == 'image'):
            transform = transforms.Compose([transforms.Resize(params.imageSize),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ColorJitter(brightness=2, contrast=2, saturation=2, hue=0.1),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
            dataset = dset.ImageFolder(root = dataroot+'train/', transform = transform)
        elif (name == 'ucf101'):
            transformVideo = transforms.Compose([transforms.Resize(size = (imageSize, imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'train/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transformVideo)
        elif (name == 'hmdb51'):
            transformVideo = transforms.Compose([transforms.Resize(size = (imageSize, imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'train/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transformVideo)
        elif (name == 'ballDrop3'):
            transformVideo = transforms.Compose([transforms.Resize(size = (imageSize, imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'train/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transformVideo)
    else:
        if (name == 'mnist'):
            transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset = dset.MNIST(root = dataroot, train = False, download = True, transform = transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
            params.nc = 1
            params.nClass = 10
        elif (name == 'f_mnist'):
            transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            dataset = dset.FashionMNIST(root = dataroot, train = False, download = True, transform = transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
            params.nc = 1
            params.nClass = 10
        elif (name == 'cifar10'):
            transform = transforms.Compose([transforms.Resize(params.imageSize), transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset = dset.CIFAR10(root = dataroot, train = False, download = True, transform = transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
            params.nClass = 10
        elif (name == 'image'):
            dataset = dset.ImageFolder(root = dataroot+'test/', transform = transforms.Compose([transforms.Resize(params.imageSize),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
        elif (name == 'ucf101'):
            transformVideo = transforms.Compose([transforms.Resize(size = (imageSize, imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'test/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transformVideo)
        elif (name == 'hmdb51'):
            transformVideo = transforms.Compose([transforms.Resize(size = (imageSize, imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'test/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transformVideo)
        elif (name == 'ballDrop3'):
            transformVideo = transforms.Compose([transforms.Resize(size = (imageSize, imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'test/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transformVideo)
    dataloader = data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
    return dataloader
