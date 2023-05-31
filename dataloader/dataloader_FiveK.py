import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2
import os, shutil
import torchvision.transforms as F
import torchvision

# random.seed(1143)

def div_data(img_path):
    input_list = glob.glob(os.path.abspath(img_path + '/wei_1/*'))

    input_val = random.sample(input_list, len(input_list) // 6)
    input_train = list(set(input_list) - set(input_val))
    for data in input_val:
        shutil.copy(data, img_path + '/Inputs_jpg_val/')
        data = data.replace('wei_1', 'wei_GT')
        data = data.replace('_1_1', '_2_1')
        shutil.copy(data, img_path + '/Experts_C_val/')
    for data in input_train:
        shutil.copy(data, img_path + '/Inputs_jpg_train/')
        data = data.replace('wei_1', 'wei_GT')
        data = data.replace('_1_1', '_2_1')
        shutil.copy(data, img_path + '/Experts_C_train/')

    return


def get_image_index(path):
    return int(path.split('.')[0].split('/')[-1])


def populate_train_list_gray(images_path):
    input_list = glob.glob(os.path.abspath(images_path + '/input/*'))
    gt_list = list(map(
        lambda x: x.replace('input', 'target'),
        input_list))
    return list(input_list), list(gt_list)

def populate_train_list_rgb(images_path):
    input_list = glob.glob(os.path.abspath(images_path + '/low/*'))
    gt_list = list(map(
        lambda x: x.replace('low', 'high'),
        input_list))
    return list(input_list), list(gt_list)



def populate_train_list_v2(images_path):
    input_list = glob.glob(os.path.abspath(images_path + '/Low/*'))
    gt_list = list(map(
        lambda x: x.replace('Low', 'Normal'),
        input_list))
    gt_list = list(map(
        lambda x: x.replace('low', 'normal'),
        gt_list))
    return list(input_list), list(gt_list)


def populate_train_list_mit(images_path):
    input_list = glob.glob(os.path.abspath(images_path + '/input/*'))
    gt_list = list(map(
        lambda x: x.replace('input', 'target'),
        input_list))
    return list(input_list), list(gt_list)



def populate_UPE_train_test(images_path,mode ="train"):

    if mode == "train":
        input_file_path = os.path.join(images_path ,"train_input.txt")
        label_file_path = os.path.join(images_path ,"train_label.txt")

        with open(input_file_path, "r") as input_file:
            input_lines = [line.strip()+".jpg" for line in input_file.readlines()]

        with open(label_file_path, "r") as label_file:
            label_lines = [line.strip()+".jpg" for line in label_file.readlines()]

        # Combine the lines from both files into a single list
        combined_list = input_lines + label_lines

        combined_list = [os.path.join(images_path,"input",line) for line in combined_list]
        gt_list = [x.replace("input","expertC") for x in combined_list]
    else :
        input_file_path = os.path.join(images_path, "test.txt")

        with open(input_file_path, "r") as input_file:
            input_lines = [line.strip()+".jpg" for line in input_file.readlines()]

        combined_list = [os.path.join(images_path, "input", line) for line in input_lines]
        gt_list = [x.replace("input", "expertC") for x in combined_list]


    return combined_list, gt_list


def RGBtoYUV444(rgb):
    # code from Jun
    # yuv range: y[0,1], uv[-0.5, 0.5]
    height, width, ch = rgb.shape
    assert ch == 3, 'rgb should have 3 channels'

    rgb2yuv_mat = np.array([[0.299, 0.587, 0.114], [-0.16874, -
    0.33126, 0.5], [0.5, -0.41869, -0.08131]], dtype=np.float32)

    rgb_t = rgb.transpose(2, 0, 1).reshape(3, -1)
    yuv = rgb2yuv_mat @ rgb_t
    yuv = yuv.transpose().reshape((height, width, 3))

    # return yuv.astype(np.float32)
    # rescale uv to [0,1]
    yuv[:, :, 1] += 0.5
    yuv[:, :, 2] += 0.5
    return yuv


class EnhanceDataset_Gray(data.Dataset):

    def __init__(self, images_path, image_size, image_size_w = None, resize=True):

        self.input_list, self.gt_list = populate_train_list_gray(images_path)
        self.image_size = image_size
        image_size_w = image_size if image_size_w is None else image_size_w
        self.image_size_w = image_size_w
        self.resize = resize
        # import pdb; pdb.set_trace()
        print("Total training examples:", len(self.input_list))

    def __getitem__(self, index):
        data_input = cv2.imread(self.input_list[index], cv2.IMREAD_UNCHANGED)
        data_gt = cv2.imread(self.gt_list[index], cv2.IMREAD_UNCHANGED)

        # 原论文中，使用的是训练时resize到256，256 image_ds 缩放的，在train文件中
        data_input = (np.asarray(data_input[..., ::-1]) / 255.0)
        data_gt = (np.asarray(data_gt[..., ::-1]) / 255.0)

        # 返回的是单色图
        data_input = np.expand_dims(data_input, axis=0)
        data_gt = np.expand_dims(data_gt, axis=0)

        data_input = torch.from_numpy(data_input).float()
        data_gt = torch.from_numpy(data_gt).float()
        return data_input, data_gt, self.input_list[index],self.gt_list[index]

    def __len__(self):
        return len(self.input_list)
    
    
class EnhanceDataset_LOL(data.Dataset):

    def __init__(self, images_path, image_size, image_size_w = None, resize=True):

        self.input_list, self.gt_list = populate_train_list_rgb(images_path)
        self.image_size = image_size
        image_size_w = image_size if image_size_w is None else image_size_w
        self.image_size_w = image_size_w
        self.resize = resize
        # import pdb; pdb.set_trace()
        print("Total training examples:", len(self.input_list))

    def __getitem__(self, index):
        data_input = cv2.imread(self.input_list[index], cv2.IMREAD_UNCHANGED)
        data_gt = cv2.imread(self.gt_list[index], cv2.IMREAD_UNCHANGED)

        # 原论文中，使用的是训练时resize到256，256 image_ds 缩放的，在train文件中
        data_input = (np.asarray(data_input[..., ::-1]) / 255.0)
        data_gt = (np.asarray(data_gt[..., ::-1]) / 255.0)

        inp_img = torch.from_numpy(data_input).float().permute(2,0,1)
        tar_img = torch.from_numpy(data_gt).float().permute(2,0,1)

        return inp_img, tar_img, self.input_list[index],self.gt_list[index]

    def __len__(self):
        return len(self.input_list)

    
    
    
    

class EnhanceDataset_LOLv2(data.Dataset):

    def __init__(self, images_path, image_size, image_size_w = None, resize=True):

        self.input_list, self.gt_list = populate_train_list_v2(images_path)
        self.image_size = image_size
        image_size_w = image_size if image_size_w is None else image_size_w
        self.image_size_w = image_size_w
        self.resize = resize
        print("Total training examples:", len(self.input_list))

    def __getitem__(self, index):
        data_input = cv2.imread(self.input_list[index], cv2.IMREAD_UNCHANGED)
        data_gt =    cv2.imread(self.gt_list[index], cv2.IMREAD_UNCHANGED)

        # 原论文中，使用的是训练时resize到256，256 image_ds 缩放的，在train文件中
        data_input = (np.asarray(data_input[..., ::-1]) / 255.0)
        data_gt = (np.asarray(data_gt[..., ::-1]) / 255.0)

        inp_img = torch.from_numpy(data_input).float().permute(2,0,1)
        tar_img = torch.from_numpy(data_gt).float().permute(2,0,1)

        return inp_img, tar_img, self.input_list[index],self.gt_list[index]

    def __len__(self):
        return len(self.input_list)




# class EnhanceDataset_UPE(data.Dataset):
#
#     def __init__(self, images_path, image_size, image_size_w=None, resize=True,mode="train"):
#         self.image_size = image_size
#         image_size_w = image_size if image_size_w is None else image_size_w
#         self.image_size_w = image_size_w
#         self.resize = resize
#         self.mode = mode
#         self.input_list, self.gt_list = populate_UPE_train_test(images_path,mode =self.mode)
#         print("Total training examples:", len(self.input_list))
#
#     def __getitem__(self, index):
#         data_input = cv2.imread(self.input_list[index], cv2.IMREAD_UNCHANGED)
#         data_gt = cv2.imread(self.gt_list[index], cv2.IMREAD_UNCHANGED)
#
#         data_input = (np.asarray(data_input[..., ::-1]) / 65535.0)
#         data_gt = (np.asarray(data_gt[..., ::-1]) / 255.0)
#
#         inp_img = torch.from_numpy(data_input).float().permute(2, 0, 1)
#         tar_img = torch.from_numpy(data_gt).float().permute(2, 0, 1)
#
#         if inp_img.shape[1] > inp_img.shape[2]:
#             inp_img = torch.rot90(inp_img, k=1, dims=(1, 2))
#             tar_img = torch.rot90(tar_img, k=1, dims=(1, 2))
#
#         _resize = F.Resize((448,600), interpolation=F.InterpolationMode.BICUBIC)
#
#         if self.resize == True:
#             inp_img = _resize(inp_img)
#             tar_img = _resize(tar_img)
#
#         torchvision.utils.save_image(inp_img,'./output/{}_out_2.png'.format(index))
#         torchvision.utils.save_image(tar_img, './output/{}_gt_2.png'.format(index))
#
#         return inp_img, tar_img, self.input_list[index], self.gt_list[index]
#
#     def __len__(self):
#         return len(self.input_list)





class EnhanceDataset_UPE(data.Dataset):

    def __init__(self, images_path, image_size, image_size_w=None, resize=True,mode="train"):
        self.image_size = image_size
        image_size_w = image_size if image_size_w is None else image_size_w
        self.image_size_w = image_size_w
        self.resize = resize
        self.mode = mode
        self.input_list, self.gt_list = populate_UPE_train_test(images_path,mode =self.mode)
        print("Total training examples:", len(self.input_list))

    def __getitem__(self, index):
        data_input = cv2.imread(self.input_list[index], cv2.IMREAD_UNCHANGED)
        data_gt = cv2.imread(self.gt_list[index], cv2.IMREAD_UNCHANGED)

        data_input = (np.asarray(data_input[..., ::-1]) / 255.0)
        data_gt = (np.asarray(data_gt[..., ::-1]) / 255.0)

        inp_img = torch.from_numpy(data_input).float().permute(2, 0, 1)
        tar_img = torch.from_numpy(data_gt).float().permute(2, 0, 1)

        if inp_img.shape[1] > inp_img.shape[2]:
            inp_img = torch.rot90(inp_img, k=1, dims=(1, 2))
            tar_img = torch.rot90(tar_img, k=1, dims=(1, 2))

        _resize = F.Resize((448,600), interpolation=F.InterpolationMode.BICUBIC)

        if self.resize == True:
            inp_img = _resize(inp_img)
            tar_img = _resize(tar_img)

        torchvision.utils.save_image(inp_img,'./output/{}_out_2.png'.format(index))
        torchvision.utils.save_image(tar_img, './output/{}_gt_2.png'.format(index))

        return inp_img, tar_img, self.input_list[index], self.gt_list[index]

    def __len__(self):
        return len(self.input_list)









class EnhanceDataset_MIT(data.Dataset):

    def __init__(self, images_path, image_size, image_size_w=None, resize=True):
        self.input_list, self.gt_list = populate_train_list_mit(images_path)
        self.image_size = image_size
        image_size_w = image_size if image_size_w is None else image_size_w
        self.image_size_w = image_size_w
        self.resize = resize
        print("Total training examples:", len(self.input_list))

    def __getitem__(self, index):
        data_input = cv2.imread(self.input_list[index], cv2.IMREAD_UNCHANGED)
        data_gt = cv2.imread(self.gt_list[index], cv2.IMREAD_UNCHANGED)

        # 原论文中，使用的是训练时resize到256，256 image_ds 缩放的，在train文件中

        data_input = (np.asarray(data_input[..., ::-1]) / 255.0)
        data_gt = (np.asarray(data_gt[..., ::-1]) / 255.0)

        inp_img = torch.from_numpy(data_input).float().permute(2, 0, 1)
        tar_img = torch.from_numpy(data_gt).float().permute(2, 0, 1)

        if inp_img.shape[1] > inp_img.shape[2]:
            inp_img = torch.rot90(inp_img, k=1, dims=(1, 2))
            tar_img = torch.rot90(tar_img, k=1, dims=(1, 2))

        _resize = F.Resize((400, 600), interpolation=F.InterpolationMode.BICUBIC)

        if self.resize == True:
            inp_img = _resize(inp_img)
            tar_img = _resize(tar_img)

        # torchvision.utils.save_image(inp_img,'./output/{}_out_2.png'.format(index))
        # torchvision.utils.save_image(tar_img, './output/{}_gt_2.png'.format(index))

        return inp_img, tar_img, self.input_list[index], self.gt_list[index]

    def __len__(self):
        return len(self.input_list)



        #
# if __name__ == '__main__':
#     img_path = r'D:\BaiduNetdiskDownload\datawei'
#     li = div_data(img_path)
#     print(li)
