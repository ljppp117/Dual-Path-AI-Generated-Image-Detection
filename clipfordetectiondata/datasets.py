# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import io
import torch

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from clipforfakedetection.clipfordetectiondata.patchselect import patch_img
from kornia.augmentation import RandomJPEG, RandomGaussianBlur
import random

"""

    如果在尝试导入InterpolationMode时发生ImportError（可能是因为使用的torchvision版本较低，没有这个类），则执行except块。
    在except块中，它将BICUBIC设置为Image.BICUBIC，这是PIL库中用于指定双三次插值的常量。

 """
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
"""kornia.augmentation模块包含了多种图像增强方法，这些方法可以用来增加图像数据集的多样性，提高模型的泛化能力。一些常见的图像增强操作包括：

    随机旋转
    随机平移
    随机缩放
    随机裁剪
    颜色抖动
    随机高斯模糊
    JPEG噪声
"""
import kornia.augmentation as K

"""加噪声和jpeg压缩"""
Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)
jpeg_95 = RandomJPEG(jpeg_quality=(95, 95), p=1.0)
jpeg_90 = RandomJPEG(jpeg_quality=(90, 90), p=1.0)
jpeg_75 = RandomJPEG(jpeg_quality=(75, 75), p=1.0)
jpeg_50 = RandomJPEG(jpeg_quality=(50, 50), p=1.0)
# 定义特定的高斯模糊
blur_1_0 = RandomGaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0), p=1.0)
blur_2_0 = RandomGaussianBlur(kernel_size=(3, 3), sigma=(2.0, 2.0), p=1.0)
blur_3_0 = RandomGaussianBlur(kernel_size=(3, 3), sigma=(3.0, 3.0), p=1.0)
# """
# 预处理流程可能用于训练阶段的图像增强，以增加数据的多样性和模型的鲁棒性
# 转换成torch.tensor 使用前面定义的随机高斯噪声和jpeg压缩
# """
# transform_before = transforms.Compose([
#     transforms.Lambda(lambda x: x.convert('RGB')),  # 确保图像是RGB格式
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: Perturbations(x.unsqueeze(0))[0])  # 应用Perturbations，注意unsqueeze(0)和squeeze(0)
# ])
# 检查Perturbations是否会导致错误的包装器
def safe_perturbations(x):
    try:
        return Perturbations(x)
    except RuntimeError as e:
        print(f"Error applying perturbations: {e}")
        return x  # 如果出现错误，返回原始图像

# 更新transform_before以使用safe_perturbations
transform_before = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),  # 确保图像是RGB格式
    transforms.ToTensor(),  # 将PIL图像转换为Tensor
    transforms.Lambda(lambda x: safe_perturbations(x.unsqueeze(0)).squeeze(0))  # 安全地应用Perturbations
])
transform_before1 = transforms.Compose([
    transforms.ToTensor()  # 将PIL图像转换为Tensor
])
"""
这个预处理流程可能用于测试阶段的图像处理，通常在测试阶段不进行数据增强，只进行基本的转换
"""
transform_before_test = transforms.Compose([
    transforms.ToTensor(),
]
)
transform_after_read = transforms.Compose([
    transforms.Resize((256, 256)),  # 将图像大小调整到256x256
    transforms.RandomCrop(224),     # 随机裁剪到224x224
])

"""这个预处理流程专门用于训练阶段，包括调整图像大小和标准化 *我这里在考虑一下是用裁剪还是resize"""
transform_train = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
"""这个预处理流程用于测试阶段，只包括标准化步骤，以确保测试数据的格式与训练数据一致"""
transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

# 定义四个不同的预处理流程
transform_before_test_jpeg95 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: jpeg_95(x.unsqueeze(0)).squeeze(0)),  # 应用JPEG压缩 (QF=95)
])

transform_before_test_jpeg90 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: jpeg_90(x.unsqueeze(0)).squeeze(0)),  # 应用JPEG压缩 (QF=90)
])

transform_before_test_jpeg75 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: jpeg_75(x.unsqueeze(0)).squeeze(0)),  # 应用JPEG压缩 (QF=75)
])
transform_before_test_jpeg50 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: jpeg_50(x.unsqueeze(0)).squeeze(0)),  # 应用JPEG压缩 (QF=50)
])

transform_before_test_blur1_0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: blur_1_0(x.unsqueeze(0)).squeeze(0)),  # 应用高斯模糊 (σ=1.0)
])

transform_before_test_blur2_0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: blur_2_0(x.unsqueeze(0)).squeeze(0)),  # 应用高斯模糊 (σ=2.0)
])
transform_before_test_blur3_0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: blur_3_0(x.unsqueeze(0)).squeeze(0)),  # 应用高斯模糊 (σ=3.0)
])

"""加载genimage训练数据集"""

# 只要是genimage/train or val/生成器类型/1 和 0 就可以
class TrainDataset(Dataset):
    def __init__(self, is_train, args):

        root = args['data_path'] if is_train else args['eval_data_path']

        self.data_list = []

        if 'GenImage' in root and root.split('/')[-1] != 'train':
            file_path = root

            if '0_real' not in os.listdir(file_path):
                for folder_name in os.listdir(file_path):

                    assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                    for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                        self.data_list.append(
                            {"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label": 0})

                    for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                        self.data_list.append(
                            {"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label": 1})

            else:
                for image_path in os.listdir(os.path.join(file_path, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label": 0})
                for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label": 1})
        else:

            for filename in os.listdir(root):

                file_path = os.path.join(root, filename)

                if '0_real' not in os.listdir(file_path):
                    for folder_name in os.listdir(file_path):

                        assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                        for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                            self.data_list.append(
                                {"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label": 0})

                        for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                            self.data_list.append(
                                {"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label": 1})

                else:
                    for image_path in os.listdir(os.path.join(file_path, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label": 0})
                    for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label": 1})

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):

        sample = self.data_list[index]

        image_path, targets = sample['image_path'], sample['label']
        # 得到路劲和标签

        try:
            image = Image.open(image_path).convert('RGB')
            # 转成rgb模式
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        try:
            x_min,x_max=patch_img(image)

        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))


        image = transform_before(image)  # 添加高斯噪声和jpeg压缩
        x_min = transform_before(x_min)
        x_max = transform_before(x_max)



        # 下面标准化 四个图像分别是三个噪声和一个全图
        x_0 = transform_train(image)
        x_max = transform_train(x_max)
        x_min = transform_train(x_min)




        # 返回图像和标签

        return torch.stack([x_max,x_min,x_0], dim=0), torch.tensor(int(targets))


"""寻找 '0_real' 和 '1_fake' 文件夹中的图像用作测试数据集 """


# class TestDataset(Dataset):
#     def __init__(self, is_train, args):
#
#         root = args['data_path'] if is_train else args['eval_data_path']
#
#         self.data_list = []
#
#         file_path = root
#
#         if '0_real' not in os.listdir(file_path):
#             for folder_name in os.listdir(file_path):
#
#                 assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']
#
#                 for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
#                     self.data_list.append(
#                         {"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label": 0})
#
#                 for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
#                     self.data_list.append(
#                         {"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label": 1})
#
#         else:
#             for image_path in os.listdir(os.path.join(file_path, '0_real')):
#                 self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label": 0})
#             for image_path in os.listdir(os.path.join(file_path, '1_fake')):
#                 self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label": 1})
#
#
#     def __len__(self):
#         return len(self.data_list)
#
#     def __getitem__(self, index):
#
#         sample = self.data_list[index]
#
#         image_path, targets = sample['image_path'], sample['label']
#
#         image = Image.open(image_path).convert('RGB')
#         x_min, x_max = patch_img(image)
#         image = transform_before_test(image)
#
#         x_min = transform_before_test(x_min)
#         x_max = transform_before_test(x_max)
#
#         # x_max, x_min, x_max_min, x_minmin = self.dct(image)
#
#
#
#         x_0 = transform_train(image)
#         x_max = transform_train(x_max)
#         x_min = transform_train(x_min)
#
#         return torch.stack([x_max,x_min, x_0], dim=0), torch.tensor(int(targets))

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        root = args['data_path'] if is_train else args['eval_data_path']
        self.data_list = []
        file_path = root

        if '0_real' not in os.listdir(file_path):
            for folder_name in os.listdir(file_path):
                assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                    self.data_list.append({
                        "image_path": os.path.join(file_path, folder_name, '0_real', image_path),
                        "label": 0,
                        "folder": folder_name
                    })

                for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                    self.data_list.append({
                        "image_path": os.path.join(file_path, folder_name, '1_fake', image_path),
                        "label": 1,
                        "folder": folder_name
                    })
        else:
            for image_path in os.listdir(os.path.join(file_path, '0_real')):
                self.data_list.append({
                    "image_path": os.path.join(file_path, '0_real', image_path),
                    "label": 0,
                    "folder": '0_real'
                })
            for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                self.data_list.append({
                    "image_path": os.path.join(file_path, '1_fake', image_path),
                    "label": 1,
                    "folder": '1_fake'
                })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, targets, folder = sample['image_path'], sample['label'], sample['folder']

        image = Image.open(image_path).convert('RGB')
        x_min,x_max=patch_img(image)
        image = transform_before_test(image)

        x_min = transform_before(x_min)
        x_max = transform_before(x_max)


        x_0 = transform_train(image)
        x_max = transform_train(x_max)
        x_min = transform_train(x_min)


        return torch.stack([x_max, x_min, x_0], dim=0), torch.tensor(int(targets)), folder


class TestDataset1(Dataset):
    def __init__(self, is_train, args):
        is_train=False
        root = args['data_path'] if is_train else args['eval_data_path']

        self.data_list = []
        self.folder_names = []

        file_path = root

        if '0_real' not in os.listdir(file_path):
            for folder_name in os.listdir(file_path):
                assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']
                self.folder_names.append(folder_name)

                for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                    self.data_list.append(
                        {"image_path": os.path.join(file_path, folder_name, '0_real', image_path),
                         "label": 0, "folder": folder_name})

                for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                    self.data_list.append(
                        {"image_path": os.path.join(file_path, folder_name, '1_fake', image_path),
                         "label": 1, "folder": folder_name})
        else:
            for image_path in os.listdir(os.path.join(file_path, '0_real')):
                self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label": 0, "folder": ''})
            for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label": 1, "folder": ''})


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]

        image_path, targets, folder = sample['image_path'], sample['label'], sample['folder']

        image = Image.open(image_path).convert('RGB')
        # 假设 transform_before_test 和 transform_train 是您已经定义好的图像预处理函数
        x_min, x_max = patch_img(image)
        image = transform_before_test(image)

        x_min = transform_before_test(x_min)
        x_max =transform_before_test(x_max)

        # x_max, x_min, x_max_min, x_minmin = self.dct(image)

        x_0 = transform_train(image)
        x_max = transform_train(x_max)
        x_min = transform_train(x_min)

        return torch.stack([x_max,x_min, x_0], dim=0), torch.tensor(int(targets)), folder  # 返回图像，标签和文件夹名称






