from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def canny_edge_count(patch):
    # 将PIL图像转换为OpenCV图像格式
    patch_cv = np.array(patch)[:, :, ::-1]  # 从RGB转换为BGR
    # 应用Canny边缘检测
    edges = cv2.Canny(patch_cv, 50, 150)
    # 计算边缘数量
    edge_count = np.sum(edges > 0)
    return edge_count


def patch_img(img, step_size=56):
    img_width, img_height = img.size
    step_size = int(step_size)
    num_patches_x = (img_width + step_size - 1) // step_size  # 计算需要多少个patch
    num_patches_y = (img_height + step_size - 1) // step_size
    patch_list = []

    # 遍历图像，划分为多个patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # 计算patch的坐标
            x1 = j * step_size
            y1 = i * step_size
            x2 = min((j + 1) * step_size, img_width)  # 确保不超过图像宽度
            y2 = min((i + 1) * step_size, img_height)  # 确保不超过图像高度
            # 提取patch
            patch = img.crop((x1, y1, x2, y2))

            # 调整patch尺寸到step_size x step_size
            if (x2 - x1) < step_size or (y2 - y1) < step_size:
                # 只有当patch的宽度或高度小于step_size时才进行调整
                patch = patch.resize((step_size, step_size), Image.ANTIALIAS)

            # 计算边缘数量并添加到列表
            patch_list.append((patch, canny_edge_count(patch)))

    # 根据边缘数量排序patch
    patch_list.sort(key=lambda x: x[1])

    # 返回边缘数量最少的patch和最多的patch
    new_img, _ = patch_list[0]
    last_img, _ = patch_list[-1]

    return new_img, last_img
    #原始版本

# def patch_img(img, step_size=28):
#     img_width, img_height = img.size
#     step_size = int(step_size)
#     size1 = int(step_size)*2
#     size2 = int(step_size)
#     num_patches_x = (img_width + step_size - 1) // step_size  # 计算需要多少个patch
#     num_patches_y = (img_height + step_size - 1) // step_size
#     patch_list = []
#
#     # 遍历图像，划分为多个patch
#     for i in range(num_patches_y):
#         for j in range(num_patches_x):
#             # 计算patch的坐标
#             x1 = j * step_size
#             y1 = i * step_size
#             x2 = min((j + 1) * step_size, img_width)  # 确保不超过图像宽度
#             y2 = min((i + 1) * step_size, img_height)  # 确保不超过图像高度
#             # 提取patch
#             patch = img.crop((x1, y1, x2, y2))
#             # 计算边缘数量并添加到列表
#             patch_list.append((patch, canny_edge_count(patch)))
#
#     # 根据边缘数量排序patch
#     patch_list.sort(key=lambda x: x[1])
#
#     # 提取边缘数量最少的四个patch和最多的四个patch
#     least_edge_patches = [patch_list[i][0] for i in range(4)]  # 边缘数量最少的四个patch
#     most_edge_patches = [patch_list[-i-1][0] for i in range(4)]  # 边缘数量最多的四个patch
#
#     # 创建两张新的112x112的图像
#     new_img1 = Image.new('RGB', (size1, size1))
#     new_img2 = Image.new('RGB', (size1, size1))
#
#     # 将最少的四个patch组合到一张112x112的图像中
#     new_img1.paste(least_edge_patches[0], (0, 0))  # 左上角
#     new_img1.paste(least_edge_patches[3], (size2, 0))  # 右上角
#     new_img1.paste(least_edge_patches[2], (0, size2))  # 左下角
#     new_img1.paste(least_edge_patches[1], (size2, size2))  # 右下角
#
#     # 将最多的四个patch组合到另一张112x112的图像中
#     new_img2.paste(most_edge_patches[0], (0, 0))  # 左上角
#     new_img2.paste(most_edge_patches[3], (size2, 0))  # 右上角
#     new_img2.paste(most_edge_patches[2], (0, size2))  # 左下角
#     new_img2.paste(most_edge_patches[1], (size2, size2))  # 右下角
#
#     return new_img1, new_img2

# def patch_img(img, step_size=56):
#     img_width, img_height = img.size
#     step_size = int(step_size)
#     size1 = int(step_size) * 2
#     size2 = int(step_size)
#     num_patches_x = (img_width + step_size - 1) // step_size  # 计算需要多少个patch
#     num_patches_y = (img_height + step_size - 1) // step_size
#     patch_list = []
#
#     # 遍历图像，划分为多个patch
#     for i in range(num_patches_y):
#         for j in range(num_patches_x):
#             # 计算patch的坐标
#             x1 = j * step_size
#             y1 = i * step_size
#             x2 = min((j + 1) * step_size, img_width)  # 确保不超过图像宽度
#             y2 = min((i + 1) * step_size, img_height)  # 确保不超过图像高度
#             # 提取patch
#             patch = img.crop((x1, y1, x2, y2))
#             # 计算边缘数量并添加到列表
#             patch_list.append((patch, canny_edge_count(patch)))
#
#     # 根据边缘数量排序patch
#     patch_list.sort(key=lambda x: x[1])
#
#     # 提取边缘数量最少的两个patch和最多的两个patch
#     least_edge_patches = [patch_list[i][0] for i in range(2)]  # 边缘数量最少的两个patch
#     most_edge_patches = [patch_list[-i-1][0] for i in range(2)]  # 边缘数量最多的两个patch
#
#     # 创建一个新的112x112的图像
#     new_img = Image.new('RGB', (size1, size1))
#
#     # 将最少的两个patch和最多的两个patch组合到112x112的图像中
#     new_img.paste(least_edge_patches[0], (0, 0))  # 左上角
#     new_img.paste(most_edge_patches[0], (size2, 0))  # 右上角
#     new_img.paste(least_edge_patches[1], (0, size2))  # 左下角
#     new_img.paste(most_edge_patches[1], (size2, size2))  # 右下角
#
#     return new_img
# # 读取图像
# img = Image.open('./')
#
# # 选择边缘数量最少的patch作为new_image
# new_img, last_img = patch_img(img,step_size=32)
#
# # 显示结果
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(new_img)
# plt.title('New Image (Least Edges)')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(last_img)
# plt.title('Last Image (Most Edges)')
# plt.axis('off')
#
# plt.show()