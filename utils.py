import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import torch

def makeFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rename(inPath, outPath):
    os.rename(inPath, outPath)

def copy(inPath, outPath):
    shutil.copyfile(inPath, outPath)

def sort(nameList, numKey):
    nameList.sort(key = numKey, reverse = false)

def enumerateFiles(path, prefix='image_', type = 'jpg'):
    inImage = path + prefix + '*.' + type
    for img in sorted(glob.glob(inImage), key = numericalSort):
        print(img)

def crop(inPath, coords, savedLocation):
    imageObj = Image.open(inPath)
    croppedImage = imageObj.crop(coords)
    croppedImage.save(savedLocation)

def readText(path):
    data = []
    with open(path, 'r') as f:
        for row in f:
            row = row.split(',')
            rowData = []
            for i in range(len(row)):
                rowData.append(row[i].strip())
            data.append(rowData)
    return data

def showLoss(trnLosses, testLosses):
    plt.figure(figsize = (10,5))
    plt.title("Loss During Training & Testing")
    plt.plot(trnLosses,  label = 'Train')
    plt.plot(testLosses, label = 'Test')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def saveCkpt(D, opt, epoch, params):
    resultDir = params.path + params.datasetName + '/result/'
    makeFolder(resultDir)
    resultDir = resultDir + 'CNN/'
    makeFolder(resultDir)
    modelDir = resultDir + 'model/'
    makeFolder(modelDir)
    epoch = epoch + 1
    torch.save(D, os.path.join(modelDir, '{:05d}_model.pth'.format(epoch)))
    torch.save(opt, os.path.join(modelDir, '{:05d}_optimizer.pth'.format(epoch)))

def loadCkpt(params):
    epoch = params.startEpoch
    device = torch.device("cuda:0" if (torch.cuda.is_available() and params.nGPU > 0) else "cpu")
    modelDir = params.path + params.datasetName + '/result/CNN/models'
    name = os.path.join(modelDir, '{:05d}_model.pth'.format(epoch))
    nameOpt = os.path.join(modelDir, '{:05d}_optimizer.pth'.format(epoch))
    D = torch.load(name).to(device)
    optD = torch.load(nameOpt)
    return D, optD
