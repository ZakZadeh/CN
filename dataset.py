import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from videoReader import VideoFolder, videoLoader

def laod(params, mode):
    dataroot = params.path + params.datasetName + '/'
    nWorkers = 3
    imageSize = params.imageSize
    nFrames = 4
    if (mode == 'train'):
        if (params.datasetName == 'mnist'):
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
        elif (name == 'chs2d'):
            imageSize = 64
            dataset = dset.ImageFolder(root = dataroot+'train/', transform = transforms.Compose([transforms.Resize(params.imageSize),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ColorJitter(brightness=2, contrast=2, saturation=2, hue=0.1),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
        elif (name == 'kth'):
            transformVideo = transforms.Compose([transforms.Resize(size = (params.imageSize, params.imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'train/', video_ext = '.png', nframes = nFrames, loader = videoLoader, transform = transformVideo)
            params.nc = 1
            params.inMode = 3
            params.nClass = 6
        elif (name == 'hmdb'):
            params.inMode = 3
            transformVideo = transforms.Compose([transforms.Resize(size = (params.imageSize, params.imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'train/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transformVideo)
            params.nClass = 51
        elif (name == 'ucf101'):
            params.inMode = 3
            transformVideo = transforms.Compose([transforms.Resize(size = (params.imageSize, params.imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'train/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transformVideo)
            params.nClass = 101
        elif (name == 'chs3d'):
            params.inMode = 3
            transformVideo = transforms.Compose([transforms.Resize(size = (params.imageSize, params.imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'train/', video_ext = '.png', nframes = nFrames, loader = videoLoader, transform = transformVideo)
    else:
        if (params.datasetName == 'mnist'):
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
        elif (name == 'chs2d'):
            imageSize = 64
            dataset = dset.ImageFolder(root = dataroot+'test/', transform = transforms.Compose([transforms.Resize(params.imageSize),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
        elif (name == 'kth'):
            transformVideo = transforms.Compose([transforms.Resize(size = (params.imageSize, params.imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'test/', video_ext = '.png', nframes = nFrames, loader = videoLoader, transform = transformVideo)
            params.nc = 1
            params.inMode = 3
            params.nClass = 6
        elif (name == 'hmdb'):
            params.inMode = 3
            params.nClass = 51
            transformVideo = transforms.Compose([transforms.Resize(size = (params.imageSize, params.imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'test/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transformVideo)
        elif (name == 'ucf101'):
            params.inMode = 3
            params.nClass = 101
            transformVideo = transforms.Compose([transforms.Resize(size = (params.imageSize, params.imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'test/', video_ext = '.jpg', nframes = nFrames, loader = videoLoader, transform = transformVideo)
        elif (name == 'chs3d'):
            params.inMode = 3
            transformVideo = transforms.Compose([transforms.Resize(size = (params.imageSize, params.imageSize), interpolation = Image.NEAREST), transforms.ToTensor(),])
            dataset = VideoFolder(video_root = dataroot+'test/', video_ext = '.png', nframes = nFrames, loader = videoLoader, transform = transformVideo)
    dataloader = data.DataLoader(dataset, batch_size = params.nBatch, shuffle = True, num_workers = nWorkers)
    return dataloader
