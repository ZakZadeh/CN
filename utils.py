import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys
import subprocess
import torch
import shutil
import glob
import re
from sklearn.metrics import confusion_matrix, accuracy_score

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value

def makeFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rename(inPath, outPath):
    os.rename(inPath, outPath)

def copy(inPath, outPath):
    shutil.copyfile(inPath, outPath)

def sort(nameList, numKey):
    nameList.sort(key = numKey, reverse = false)

#numbers = re.compile(r'(\d+)')
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
    
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

def vid2jpg(dir_path, dst_dir_path):
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    for file_name in os.listdir(dir_path):
        if '.mp4' not in file_name:
            continue
        name, ext = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_dir_path, name)

        video_file_path = os.path.join(dir_path, file_name)
        try:
            if os.path.exists(dst_directory_path):
                if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                    subprocess.call('rm -r {}'.format(dst_directory_path), shell=True)
                    print('remove {}'.format(dst_directory_path))
                    os.mkdir(dst_directory_path)
                else:
                    continue
            else:
                os.mkdir(dst_directory_path)
        except:
            print(dst_directory_path)
            continue
        cmd = 'ffmpeg -i {} -vf scale=-1:360 {}/image_%05d.jpg'.format(video_file_path, dst_directory_path)
        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')
        
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
