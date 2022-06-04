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

def show(img):
    cv2.imshow('temp',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    # bgr__ =cv2.imread(bgr1)
    # print(src_.shape)
    # print(bgr__.shape)

    # src_ = cv2.resize(src_, (0, 0), None, .2, .2)
    # bgr_ = cv2.resize(bgr_, (0, 0), None, .25, .25)
    # print(bgr.size)
    # print(src_.shape)
    tar_h,tar_w,_= src_.shape
    # target_face,tarface_x, tarface_y, tarface_w, tarface_h=cropFaceWithPos(src_)
    # source_face,srcface_x,srcface_y,srcface_w,srcface_h= cropFaceWithPos(src__)

    # show(target_face)
    # x_diff=tarface_x-srcface_x

    src = to_tensor(src).cuda().unsqueeze(0)
    src_relit = to_tensor(src_relit).cuda().unsqueeze(0)
    bgr = to_tensor(bgr).cuda().unsqueeze(0)

    if src.size(2) <= 2048 and src.size(3) <= 2048:
        model.backbone_scale = 1/4
        model.refine_sample_pixels = 80_000
    else:
        model.backbone_scale = 1/8
        model.refine_sample_pixels = 320_000
    pha, fgr = model(src, bgr)[:2]

    com = pha * fgr + (1 - pha) * torch.tensor([0, 0, 0], device='cuda').view(1, 3, 1, 1)
    matted_img=to_pil_image(com[0].cpu())
    # com = pha * fgr + (1 - pha) * torch.tensor([120/255, 255/255, 155/255], device='cuda').view(1, 3, 1, 1)
    bgr_img = to_tensor(bgr_img).cuda().unsqueeze(0)
    com = pha * src_relit + (1 - pha) * bgr_img.clone().detach()
    # com = pha * src_relit + (1 - pha) * torch.tensor(bgr_img, device='cuda')
    # to_pil_image(com[0].cpu()).save('res.png')
    res_img=to_pil_image(com[0].cpu())
    res_cv2 = cv2.cvtColor(np.asarray(res_img), cv2.COLOR_RGB2BGR)
    show(res_cv2)
    # to_pil_image(com[0].cpu()).save('res.png')
    return res_cv2
    # to_pil_image(pha[0].cpu()).save('temp.png')
    # pha=to_pil_image(pha[0].cpu())
    # pha_cv2 = cv2.cvtColor(np.asarray(pha), cv2.COLOR_RGB2BGR)
    # ph.show()

    # matted_img.show()
    # matted_cv2 = cv2.cvtColor(np.asarray(matted_img), cv2.COLOR_RGB2BGR)
    # # matted_cv2&=pha_cv2
    # # matted_cv2 = cv2.resize(matted_cv2, (0, 0), None, .5, .5)
    # matted_h,matted_w,_=matted_cv2.shape
    # show(matted_cv2)
    # print('mat',matted_cv2.shape)
    # source_face,srcface_x,srcface_y,srcface_w,srcface_h= cropFaceWithPos(matted_cv2)
    # show(source_face)
    # print(source_face.shape)
    # print(target_face.shape)

    # face_h_ratio=srcface_h/tarface_h
    # # w_ratio=tarface_w/srcface_w
    # body_h_ratio=(matted_h-srcface_y)/(tar_h-tarface_y)
    # print(body_h_ratio, face_h_ratio)
    # h_ratio=(body_h_ratio+face_h_ratio)/2
    # print('h', h_ratio)
    # bgr_ = cv2.resize(bgr_, (0, 0), None, h_ratio, h_ratio)
    # src_ = cv2.resize(src_, (0, 0), None, h_ratio, h_ratio)
    # # show(matted_cv2)
    # source_face,srcface_x,srcface_y,srcface_w,srcface_h= cropFaceWithPos(src_)
    # tar_h,tar_w,_= bgr_.shape

    # black=np.zeros((tar_h,tar_w,3), np.uint8)
    # # black[:]=(155,255,120)


    # y_diff=tar_h-matted_h
    # x_diff=tarface_x-srcface_x
    # print(y_diff)
    # print(x_diff)

    # # for y in range(matted_h-1,0,-1):
    # #     for x in range(matted_w):
    # #         if y+y_diff>=tar_h   or x+x_diff>=tar_w:
    # #             break 
    # #         elif y+y_diff<0 or x+x_diff<0:
    # #             continue 
    # #         else:
    # #             black[y+y_diff][x+x_diff]=matted_cv2[y][x]
    # # show(black)

    # # t=np.array([155,255,120])
    # for y in range(matted_h-1,0,-1):
    #     for x in range(matted_w):
    #         if y+y_diff>=tar_h   or x+x_diff>=tar_w:
    #             break 
    #         elif matted_cv2[y][x].any():
    #             if (matted_cv2[y][x][0]+matted_cv2[y][x][1]+matted_cv2[y][x][2])>5:
    #             # print(black[y][x])
    #                 bgr_[y+y_diff][x+x_diff]= matted_cv2[y][x]
    #         # if not ((black[y][x]==t).all()):
    # # show(np.hstack((src_,bgr_)))
    # # bgr_=cv2.resize(bgr`_,bgr.size)
    # print(bgr_.shape)
    # # bgr_=cv2.medianBlur(bgr_,11)
    # show(bgr_)

    # cv2.imwrite('res.png', bgr_)
    # return bgr_

if __name__ == '__main__':
    backReplace('30.png','30.png','30_bgr.png', 'src_.png', 'bgr_.png')
    # backReplace('src.png','bgr.png', 'src_.png', 'bgr_.png')
    # backReplace('30.png','30.png','30_bgr.png', 'src.png', 'bgr.png')
    # backReplace('src.png', 'src.png', 'bgr.png','30.png','30_bgr.png')

