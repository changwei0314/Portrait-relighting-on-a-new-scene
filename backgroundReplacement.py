import sys
import os
# sys.path.append('CV-Final-Project/face_detect')
sys.path.append('face_detect')
from faceDetect import cropFaceWithPos,cropFace
import torch
import cv2
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms import ToTensor
from PIL import Image

def show(name,img):
    cv2.imshow(name,img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def backReplace(src1_relit, src1, bgr1, src2, bgr2):
    
    model = torch.jit.load('model.pth').cuda().eval()
    # print(os.path.dirname(os.path.abspath(__file__)))

    src = Image.open(src1)

    src_relit = Image.open(src1_relit)

    bgr = Image.open(bgr1)
    bgr_img = Image.open(bgr2)
    bgr_img=bgr_img.resize(bgr.size)
    src__ =cv2.imread(src1)
    src_ =cv2.imread(src2)
    bgr_ =cv2.imread(bgr2)
    src_=cv2.resize(src_, bgr.size,None)
    bgr_=cv2.resize(bgr_, bgr.size,None)

    tar_h,tar_w,_= src_.shape

    src = to_tensor(src).cuda().unsqueeze(0)
    src_relit = to_tensor(src_relit).cuda().unsqueeze(0)
    bgr = to_tensor(bgr).cuda().unsqueeze(0)
    bgr_img = to_tensor(bgr_img).cuda().unsqueeze(0)

    if src.size(2) <= 2048 and src.size(3) <= 2048:
        model.backbone_scale = 1/4
        model.refine_sample_pixels = 65000
    else:
        model.backbone_scale = 1/8
        model.refine_sample_pixels = 320_000
    pha, fgr = model(src, bgr)[:2]

    com = pha * fgr + (1 - pha) * torch.tensor([0, 0, 0], device='cuda').view(1, 3, 1, 1)
    matted_img=to_pil_image(com[0].cpu())
    # com = pha * fgr + (1 - pha) * torch.tensor([120/255, 255/255, 155/255], device='cuda').view(1, 3, 1, 1)
    com = pha * src_relit + (1 - pha) * bgr_img.clone().detach()
    # com = pha * src_relit + (1 - pha) * torch.tensor(bgr_img, device='cuda')
    # to_pil_image(com[0].cpu()).save('res.png')
    res_img=to_pil_image(com[0].cpu())
    res_cv2 = cv2.cvtColor(np.asarray(res_img), cv2.COLOR_RGB2BGR)

    src1_relit=cv2.imread(src1_relit)
    show('src1_relit',src1_relit)
    src1=cv2.imread(src1)
    show('src1',src1)
    src2=cv2.imread(src2)
    show('src2',src2)
    #bgr1=cv2.imread(bgr1)
    #show('bgr1',bgr1)
    #bgr2=cv2.imread(bgr2)
    #show('bgr2',bgr2)

    show('result',res_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # to_pil_image(com[0].cpu()).save('res.png')
    return res_cv2
 

