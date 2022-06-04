'''
    this is a simple test file
'''
import sys
sys.path.append('model')
sys.path.append('utils')
sys.path.append('face_detect')

from utils_SH import *

from faceDetect import cropFace

# other modules
import os
import numpy as np

from torch.autograd import Variable
from backgroundReplacement import backReplace
import torch
import cv2
import argparse

# This code is adapted from https://github.com/zhhoper/DPR

def parse_args():
    parser = argparse.ArgumentParser(
        description="image relighting")
    parser.add_argument(
        '--source_image',
        default='obama.jpg',
        help='name of image stored in data/test/images',
    )
    parser.add_argument(
        '--light_image',
        default='obama.jpg',
        help='name of image stored in data/test/images',
    )
    parser.add_argument(
        '--model',
        default='trained.pt',
        help='model file to use stored in trained_model/'
    )
    parser.add_argument(
        '--face_detect',
        default='Neither',
        help='Options: "both" or "light". Face detection/cropping for more accurate relighting.'
    )
    
    return parser.parse_args()

def preprocess_image(img_path, srcOrLight):
    src_img = cv2.imread(img_path)
    if (ARGS.face_detect == 'both') or (ARGS.face_detect == 'light' and srcOrLight == 2):
        src_img = cropFace(src_img)
    row, col, _ = src_img.shape
    src_img = cv2.resize(src_img, (256, 256))
    Lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB) #converts image to one color space LAB

    inputL = Lab[:,:,0] #taking only the L channel, shape = (256,256)
    inputL = inputL.astype(np.float32)/255.0 #normalise
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]  # reshape to (1,1,256,256)
    inputL = Variable(torch.from_numpy(inputL))

    return inputL, row, col, Lab


ARGS = parse_args()

modelFolder = 'trained_models/'

# load model
from model import *
my_network = HourglassNet()
my_network.load_state_dict(torch.load(os.path.join(modelFolder, ARGS.model), map_location=torch.device('cpu')))

my_network.train(False)

# folder for saving relit result
saveFolder = 'result/'
saveFolder = os.path.join(saveFolder, ARGS.model.split(".")[0])
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

# use our model
light_img, _, _, _ = preprocess_image(f'data/test/images/{ARGS.light_image}', 2)

sh = torch.zeros((1,9,1,1))

_, outputSH  = my_network(light_img, sh, 0) 

src_img, row, col, Lab = preprocess_image(f'data/test/images/{ARGS.source_image}', 1)

outputImg, _ = my_network(src_img, outputSH, 0) # L channel, torch.Size([1, 1, 256, 256]

# get the result
outputImg = outputImg[0].cpu().data.numpy() 
outputImg = outputImg.transpose((1,2,0))    
outputImg = np.squeeze(outputImg) # shape = (256,256)
outputImg = (outputImg*255.0).astype(np.uint8)  # restore to 0~255 values
Lab[:,:,0] = outputImg  # change lighting condition of source image
resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
resultLab = cv2.resize(resultLab, (col, row))
img_name, e = os.path.splitext(ARGS.source_image)
if (ARGS.face_detect == 'both'):
    img_name += "_faceDetectBoth"
if (ARGS.face_detect == 'light'):
    img_name += "_faceDetectLight"
cv2.imwrite(os.path.join(saveFolder, f'{img_name}_relit.jpg'), resultLab)

print(os.path.join(saveFolder, f'{img_name}_relit.jpg'))
relit_path=os.path.join(saveFolder, f'{img_name}_relit.jpg')
# src1_path=f'data/test/images/{ARGS.source_image}'
# src2_path=f'data/test/images/{ARGS.light_image}'
src1_path=f'{ARGS.source_image}'
src2_path=f'{ARGS.light_image}'
bgr1_path='bgr.png'
bgr2_path='30_bgr.png'
res=backReplace(relit_path,src1_path,bgr1_path,src2_path,bgr2_path)
cv2.imwrite(os.path.join(saveFolder, f'{img_name}_replace.jpg'), res)

#cv2.imwrite(os.path.join(saveFolder, f'{img_name}_L_channel.jpg'), outputImg)
#----------------------------------------------