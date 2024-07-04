# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

# import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import PIL.Image as Image
from torchvision import transforms
import torch
import cv2
import numpy as np
from skimage import io
def main():

    # loading astronaut image
    # img = skimage.data.astronaut()
    # A_path = "D:/work/pic/测试图片/110101199603010001.jpg"
    A_path = "D:/work/pic/abds.png"
    # img = Image.open(A_path).convert('RGB')
    # data_transform = transforms.Compose(
    #     [transforms.Resize((224, 224)),
    #      transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 使用matplotlib的pyplot模块显示原始图像
    # pil.ImageShow.show(A_img)
    # 将图像数据转换为张量并且进行归一化处理
    # [N, C, H, W]是张量的维度，N为批量大小（这里只有一个样本），C是通道数，H是高度，W是宽度
    # img = data_transform(img)
    # img = torch.unsqueeze(img, dim=0)
    # 使用matplotlib的pyplot模块显示原始图像
    # pil.ImageShow.show(A_img)
    # 将图像数据转换为张量并且进行归一化处理
    # [N, C, H, W]是张量的维度，N为批量
    # img.

    # img = cv2.imread(A_path)
    # img = Image.open(A_path).convert('RGB')
    # img = numpy.array(img)
    # img = img.astype(np.float32, copy=True)
    # perform selective search
    img = io.imread(A_path)

    '''selectivesearch 调用selectivesearch函数 对图片目标进行搜索
    #Parameters
    ----------
        im_orig : 类型ndarray
            Input image  
            输入图片
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
            自由参数。在felzenszwalb分割中，较高的聚类数意味着较大的聚类数。
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
            用于felzenszwalb分割的高斯核宽度。
        min_size : int
            Minimum component size for felzenszwalb segmentation.
            felzenszwalb分割的最小分量大小
    '''


    img_lbl, regions = selectivesearch.selective_search(
        img, scale=5, sigma=0.75, min_size=5)

    # 1）第一次过滤
    candidates = []
    for r in regions:
        # 重复的不要
        if r['rect'] in candidates:
            continue
        # 太小和太大的不要
        if r['size'] < 2000 or r['size'] > 20000:
            continue
        x, y, w, h = r['rect']
        # 太不方的不要
        if w / h > 1 or h / w > 2:  # 根据实际情况调整
            continue
        candidates.append((x, y, w, h))
    print('len(candidates)', len(candidates))

    # 2)第二次过滤 大圈套小圈的目标 只保留大圈
    num_array = []
    for i in candidates:
        if len(num_array) == 0:
            num_array.append(i)
        else:
            content = False
            replace = -1
            index = 0
            for j in num_array:
                ##新窗口在小圈 则滤除
                if i[0] >= j[0] and i[0] + i[2] <= j[0] + j[2] and i[1] >= j[1] and i[1] + i[3] <= j[1] + j[3]:
                    content = True
                    break
                ##新窗口不在小圈 而在老窗口外部 替换老窗口
                elif i[0] <= j[0] and i[0] + i[2] >= j[0] + j[2] and i[1] <= j[1] and i[1] + i[3] >= j[1] + j[3]:
                    replace = index
                    break
                index += 1
            if not content:
                if replace >= 0:
                    num_array[replace] = i
                else:
                    num_array.append(i)

    # 窗口过滤完之后的数量
    # 二次过滤后剩余10个窗
    print('len(candidates)', len(num_array))

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    # for x, y, w, h in candidates:
    for x, y, w, h in num_array:
        print(x, y, w, h)
        # rect = mpatches.Rectangle(
        #     (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        # ax.add_patch(rect)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    main()