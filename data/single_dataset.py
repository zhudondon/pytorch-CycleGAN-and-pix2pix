from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))
    #   这里只返回了，转换的方式列表；不进行具体的转换 transform_list = [] 返回了这个list
    #  transforms.Compose 是 PyTorch 中 torchvision.transforms 模块提供的一个类，用于将多个图像转换操作组合成一个序列。
    #  这个类允许用户定义一个转换操作的列表，然后将这个列表中的所有操作按照顺序应用到图像数据上。
    #
    # 在 PyTorch 中，图像预处理通常需要进行多个步骤，例如将图像转换为张量、归一化、随机裁剪等。
    # 使用 transforms.Compose 可以方便地将这些操作按照指定的顺序组合起来，形成一个完整的预处理流程。


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
