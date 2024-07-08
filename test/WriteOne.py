import os

import PIL.Image as Image
import numpy as np
import torch
# from util import tensor2im
import torch.nn as nn
import torch.nn.functional as func
import torchvision.utils as tutils
from torchvision import transforms

# 更改预训练下载位置
os.environ['TORCH_HOME'] = 'data/pretrained_weights/'


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs", comment='test your comment', filename_suffix=" test your filename suffix")

A_path = "D:/work/pic/测试图片/110101199603010001.jpg"
A_img = Image.open(A_path).convert('RGB')
# opt = TrainOptions().parse()

transform_list = []

transform_list.append(transforms.Resize((256, 256)))
transform_list += [transforms.ToTensor()]
transform_list += [transforms.Normalize((0.5,), (0.5,))]
# transform_list.append(transforms.Grayscale(1))


# transform = get_transformNew(grayscale=True)
# transform = transforms.Compose(transform_list)
#
# A = transform(A_img)
# A = torch.reshape(A, [1, 3, 256, 256])

channelNum = 3
# tensor = torch.FloatTensor([[0, 1, -0.5], [0, 1, -1], [0, 1, -1]])
n = 3
m = 3

times = 1
# def once_do(conv_data, n,m):
#     tensor = torch.randn((n, m), dtype=torch.float32)
#     print("tensor:", ca, tensor)
#     tensor = torch.reshape(tensor, [1, 1, n, m])
#     conv_data = func.conv2d(conv_data, tensor, stride=[1], padding=[0], dilation=[1])
#     return func.leaky_relu(conv_data, True)


# for ca in range(channelNum):
#     ca_ = A[0, ca, :, :]
#     ca_ = torch.reshape(ca_, [1, 1, 256, 256])
#     conv_data = ca_
#     for ba in range(times):
#        conv_data = once_do(conv_data, n, m)
#
#     print("conv:", ca, conv_data)
#     im = tensor2im(conv_data)
#     image_pil = Image.fromarray(im)
#     PIL.ImageShow.show(image_pil)

from torchvision.models import alexnet

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 使用matplotlib的pyplot模块显示原始图像
# pil.ImageShow.show(A_img)
# 将图像数据转换为张量并且进行归一化处理
# [N, C, H, W]是张量的维度，N为批量大小（这里只有一个样本），C是通道数，H是高度，W是宽度
img = data_transform(A_img)
# 在张量的维度上增加一个维度，以使其可以作为网络的输入
img = torch.unsqueeze(img, dim=0)


alex_net = alexnet(pretrained=True)
# alex_net.
# alex_net(img)

kernel_num = -1
vis_max = 1
for sub_mod in alex_net.modules():
    if isinstance(sub_mod, nn.Conv2d):
        kernels = sub_mod.weight
        c_out, c_in, k_w, k_h = tuple(kernels.shape)

        kernel_num += 1
        if kernel_num > vis_max:
            break

        # for o_idx in range(c_out):
        #     kernels_o_idx = kernels[o_idx, :, :, :].unsqueeze(1)
        #     grid = tutils.make_grid(kernels_o_idx, nrow=c_in, normalize=True, scale_each=True)
        #     writer.add_image('{} split in channel'.format(o_idx), grid, global_step=322)

        kernel_all = kernels.view(-1, 3, k_h, k_w)
        grid = tutils.make_grid(kernel_all, nrow=8)
        writer.add_image('{} all '.format(str(kernels.shape)), grid, global_step=322)

features_ = alex_net.features[0]
features_res = features_(img)

features_res.transpose_(0, 1)  #bchw=(1,64,55，55)-->(64，1，55，55)
gridN = tutils.make_grid(features_res, nrow=8, normalize=True, scale_each=True)
writer.add_image('feature map in cony1', gridN, global_step=322)


writer.close()


fake_img =torch.ones((1,1,4,4))# batch size * channel * H* W
output = alex_net(fake_img)

def hookFw():
    print()
alex_net.register_forward_hook(hook= hookFw)

loss_fnc = nn.LlLoss()
target = torch.randn_like(output)
loss = loss_fnc(target, output)
loss.backward()










# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("logs")  # #第一个参数指明writer把summary内容写在哪个目录下
# # y = x
# for i in range(100):
#     writer.add_scalar("y=x", i, i)  # 添加数据
#     #  参数：图表名称，y轴，x轴
# # y = 2x
# for i in range(100):
#     writer.add_scalar("y=x", 2 * i, i)  # 添加数据
# writer.close()


# tensor = torch.FloatTensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# tensor = torch.FloatTensor([[0, 1, -0.5], [0, 1, -1], [0, 1, -1]])
# tensor = torch.rand((3, 3), dtype=torch.float32)
# tensor = torch.randint(1, 100, (3, 3), dtype=torch.float32)
# tensor = torch.tensor([[1, 2, 1], [0, 1, 0], [2, 1, 0]])

# conv_d = func.conv2d(A, tensor, stride=[1], padding=[0], dilation=[1])

# print(conv_d)
#
# # 将结果 变回图片
# im = tensor2im(conv_d)
# image_pil = Image.fromarray(im)
#
# PIL.ImageShow.show(image_pil)















