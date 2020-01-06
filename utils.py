import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys
import subprocess
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
